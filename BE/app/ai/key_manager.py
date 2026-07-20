from enum import Enum
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from app.ai.gemini import AVAI_GEMINI_MODELS    
from app.core.config import settings

logger = logging.getLogger(__name__)

# Vietnam Timezone (UTC+7)
VN_TZ = timezone(timedelta(hours=7))

class Key:
    def __init__(self, key: str, key_id: int = 1):
        self.key   = key
        self.id    = key_id                              # 1-indexed vị trí trong GEMINI_API_KEYS
        self.tier  = "free"                              # tất cả key trong GEMINI_API_KEYS là free
        self.current_error = None
        self.reset_day = None  # thời điểm reset rate limit (datetime)
        self.exhausted_models: set[str] = set()  # set các model đã cạn kiệt RPD trên key này
        self.request_count: int = 0  # số request đã gửi — dùng để log stats sau mỗi chapter

    def calculate_next_reset(self) -> datetime:
        """
        Tính thời gian reset tiếp theo theo giờ Việt Nam (UTC+7):
        - Từ tháng 4 đến tháng 10: reset vào 14:00 VN
        - Từ tháng 11 đến tháng 3 năm sau: reset vào 15:00 VN
        """
        now_vn = datetime.now(VN_TZ)
        month = now_vn.month
        
        # Xác định giờ reset theo tháng
        if 4 <= month <= 10:
            reset_hour = 14
        else:
            reset_hour = 15
            
        # Tạo thời gian reset cho ngày hôm nay
        reset_today = now_vn.replace(hour=reset_hour, minute=0, second=0, microsecond=0)
        
        if now_vn < reset_today:
            return reset_today
        else:
            # Cộng thêm 1 ngày nếu đã qua giờ reset của ngày hôm nay
            return reset_today + timedelta(days=1)

# ---------------------------------------------------------------------------
# Key Manager (Singleton)
# ---------------------------------------------------------------------------

def get_all_api_keys():
    """
    Khởi tạo danh sách Key từ settings.GEMINI_API_KEYS.
    Tất cả key này là Free tier — dùng cho Async và Sync mode.
    Pro keys (Rich Mode) được quản lý riêng qua get_pro_key_pool() trong pipeline_helpers.
    """
    return [Key(api_key, key_id=i + 1) for i, api_key in enumerate(settings.GEMINI_API_KEYS)]

class KeyManager:
    """
    Singleton quản lý pool API key với:
    - Round-robin distribution (phân tải đều giữa các user/request)
    - Daily exhaustion tracking (đánh dấu key hết quota ngày)
    - Lưu trữ trạng thái bền vững (JSON)
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.keys = get_all_api_keys()
        self._round_robin_index = 0

        logger.info(f"🔑 KeyManager khởi tạo với {len(self.keys)} API key(s).")
        self.load_state()

    def _mask_key(self, key: Key) -> str:
        """Mask API key để log an toàn: AIzaSy...Juw"""
        if len(key.key) <= 12:
            return key.key[:4] + "..."
        return key.key[:6] + "..." + key.key[-3:]

    def save_state(self):
        """Lưu trạng thái các key (reset_day, exhausted_models) xuống file JSON."""
        state_path = "keys_state.json"
        try:
            state_dict = {}
            for k in self.keys:
                state_dict[k.key] = {
                    "reset_day": k.reset_day.isoformat() if k.reset_day else None,
                    "exhausted_models": list(k.exhausted_models)
                }
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state_dict, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Lỗi khi lưu trạng thái API keys: {e}")

    def load_state(self):
        """Tải trạng thái các key từ file JSON nếu tồn tại."""
        state_path = "keys_state.json"
        if not os.path.exists(state_path):
            return
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            for k in self.keys:
                if k.key in state_dict:
                    info = state_dict[k.key]
                    if info.get("reset_day"):
                        k.reset_day = datetime.fromisoformat(info["reset_day"])
                    k.exhausted_models = set(info.get("exhausted_models", []))
        except Exception as e:
            logger.error(f"⚠️ Lỗi khi tải trạng thái API keys: {e}")

    def get_available_keys_for_models(self, model_names: list[str]) -> list[Key]:
        """Trả về danh sách đối tượng Key mà vẫn còn ít nhất 1 model trong model_names chưa hết quota."""
        now_vn = datetime.now(VN_TZ)
        available = []
        state_changed = False
        
        for k in self.keys:
            # Nếu đã qua thời điểm reset, khôi phục lại quota cho key
            if k.reset_day is not None and now_vn >= k.reset_day:
                k.reset_day = None
                k.exhausted_models.clear()
                state_changed = True
                
            # Một key khả dụng nếu có ít nhất 1 model trong model_names không nằm trong exhausted_models
            if not model_names:
                if k.reset_day is None:
                    available.append(k)
            else:
                if any(m not in k.exhausted_models for m in model_names):
                    available.append(k)
                    
        if state_changed:
            self.save_state()
            
        return available

    def get_available_keys(self) -> list[Key]:
        """Tương thích ngược: kiểm tra key chưa bị cạn kiệt hoàn toàn."""
        return self.get_available_keys_for_models([])

    def get_next_key_for_models(self, model_names: list[str]) -> Key:
        """
        Round-robin chọn Key tiếp theo có sẵn model trong model_names. 
        Đoạn code chính khiến xoay vòng key
        """
        available = self.get_available_keys_for_models(model_names)
        if not available:
            # Tìm thời điểm reset sớm nhất trong các key bị cạn kiệt cho model_names
            exhausted_keys = [k for k in self.keys if any(m in k.exhausted_models for m in model_names)]
            next_reset = min((k.reset_day for k in exhausted_keys if k.reset_day is not None), default=None)
            reset_info = f" đến {next_reset.strftime('%H:%M %d/%m/%Y')} VN" if next_reset else ""
            raise RuntimeError(
                f"🚫 TẤT CẢ API KEY ĐÃ HẾT QUOTA CHO CÁC MODEL NÀY. "
                f"Quota sẽ bắt đầu khả dụng trở lại{reset_info}. "
                f"Hãy thêm API key mới hoặc đợi."
            )
        key = available[self._round_robin_index % len(available)]
        self._round_robin_index += 1
        return key

    def get_next_key(self) -> Key:
        """Tương thích ngược: lấy key kế tiếp chưa bị cạn kiệt hoàn toàn."""
        return self.get_next_key_for_models([])

    def mark_model_exhausted(self, key: Key, model_name: str):
        """Đánh dấu model cụ thể đã hết quota trên key này."""
        key.exhausted_models.add(model_name)
        if key.reset_day is None:
            key.reset_day = key.calculate_next_reset()
            
        logger.warning(
            f"📅 Model {model_name} trên API Key {self._mask_key(key)} đã hết quota ngày (RPD). "
            f"Bỏ qua model này trên key đến {key.reset_day.strftime('%H:%M %d/%m/%Y')} VN."
        )
        self.save_state()

    def mark_daily_exhausted(self, key: Key):
        """Tương thích ngược: đánh dấu toàn bộ model hiện tại bị hết quota."""
        # Giả định đánh dấu model chính hiện tại bằng cách để trống hoặc đánh dấu toàn bộ model
        for m in AVAI_GEMINI_MODELS:
            key.exhausted_models.add(m)
        if key.reset_day is None:
            key.reset_day = key.calculate_next_reset()
        logger.warning(
            f"📅 API Key {self._mask_key(key)} đã bị đánh dấu hết quota ngày cho toàn bộ model."
        )
        self.save_state()

    def get_remaining_keys(self, current_key: Key) -> list[Key]:
        """Tương thích ngược: Trả về các Key available ngoại trừ key hiện tại."""
        available = self.get_available_keys()
        return [k for k in available if k != current_key]

    def reset_daily_tracking(self):
        """Reset tracking (chủ yếu dùng cho testing/Celery task)."""
        for k in self.keys:
            k.reset_day = None
            k.exhausted_models.clear()
        self.save_state()

# ---------------------------------------------------------------------------
# Error Classification
# ---------------------------------------------------------------------------

class ErrorType(Enum):
    """Phân loại lỗi thành 3 nhóm để quyết định hành vi fallback."""
    PER_MODEL = "per_model"           # Lỗi riêng model → skip model, giữ key
    RATE_LIMIT_RPM = "rate_limit"     # 429 RPM/TPM → chỉ đổi model, giữ key
    DAILY_EXHAUSTED = "daily"         # 429 RPD (hết quota ngày) → đánh dấu key + đổi key