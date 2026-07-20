"""
Module: app.utils.typesetting
Description: Implements optimal line breaking algorithms (using dynamic programming) and font-size calculations.
"""

from manga_translator.rendering import text_render

def split_into_balanced_lines(text: str, n_lines: int) -> list[str]:
    """
    Splits text into balanced lines using dynamic programming to minimize the maximum line length difference.
    """
    words = text.split()
    if not words:
        return [""]
    if len(words) <= n_lines:
        return words + [""] * (n_lines - len(words))
    
    memo = {}
    
    def solve(word_idx, lines_left):
        state = (word_idx, lines_left)
        if state in memo:
            return memo[state]
        
        if lines_left == 1:
            line_str = " ".join(words[word_idx:])
            return len(line_str), [line_str]
        
        if word_idx >= len(words):
            return 0, [""] * lines_left
            
        best_cost = float('inf')
        best_partition = []
        
        for i in range(word_idx + 1, len(words) - lines_left + 2):
            line_str = " ".join(words[word_idx:i])
            cost_curr = len(line_str)
            
            cost_rest, partition_rest = solve(i, lines_left - 1)
            cost = max(cost_curr, cost_rest)
            
            if cost < best_cost:
                best_cost = cost
                best_partition = [line_str] + partition_rest
                
        memo[state] = (best_cost, best_partition)
        return memo[state]
        
    _, result = solve(0, n_lines)
    return result

def find_optimal_typesetting(text: str, shrunk_width: int, shrunk_height: int) -> tuple[int, int, list[str]]:
    """
    Finds the optimal font size, number of lines, and line partitions within bounds shrunk_width and shrunk_height.
    """
    best_raw_font_size = -1
    best_n_lines = 1
    best_lines = [text]
    
    # Iterate from 1 to 5 lines to find the maximum possible font size and best aesthetic balance
    for n in range(1, 6):
        lines = split_into_balanced_lines(text, n)
        max_chars = max(len(l) for l in lines)
        if max_chars == 0:
            continue
            
        # Average width of Vietnamese/Latin characters ~ 0.55 * font_size
        font_size_width = int(shrunk_width / (max_chars * 0.55))
        # Average height of lines with 1.25 line height multiplier ~ 1.25 * font_size
        font_size_height = int(shrunk_height / (n * 1.25))
        
        raw_font_size = min(font_size_width, font_size_height)
        
        if raw_font_size > best_raw_font_size:
            best_raw_font_size = raw_font_size
            best_n_lines = n
            best_lines = lines
        elif raw_font_size == best_raw_font_size and n < best_n_lines:
            # Prefer fewer lines in case of font size ties
            best_raw_font_size = raw_font_size
            best_n_lines = n
            best_lines = lines
            
    # Keep final font size strictly within scanlation-standard safety margins [12, 26]
    final_font_size = max(12, min(best_raw_font_size, 26))
    return final_font_size, best_n_lines, best_lines
