#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to remove unused legacy functions from technical_analysis.py
"""

# Read the file
with open('core/technical_analysis.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Functions to remove (line numbers are 0-indexed)
functions_to_remove = [
    (518, 'calculate_bullish_signal_quality'),
    (557, 'calculate_bearish_signal_quality'),
    (596, 'calculate_signal_direction'),
    (614, 'get_entry_signal_label'),
    (655, 'calculate_mpi_signal_quality'),
    (701, 'calculate_mpi_position_score'),
    (724, 'calculate_ibs_confirmation_score'),
    (759, 'calculate_relvol_confirmation_score'),
    (800, 'calculate_signal_quality_v2'),
    (858, 'get_signal_label_v2'),
    (1146, 'calculate_acceleration_score'),
]

def find_function_end(lines, start_idx):
    """Find the line where the function ends"""
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith('def ') or lines[i].startswith('# ====='):
            return i
    return len(lines)

# Build list of line ranges to remove
ranges_to_remove = []
for start_line, func_name in functions_to_remove:
    end_line = find_function_end(lines, start_line)
    ranges_to_remove.append((start_line, end_line))
    print(f"Will remove {func_name}: lines {start_line+1}-{end_line} ({end_line-start_line} lines)")

# Create new file content by excluding the ranges
new_lines = []
skip_until = -1

for i, line in enumerate(lines):
    if i < skip_until:
        continue  # Skip lines in removal range
    
    # Check if this line starts a removal range
    in_removal_range = False
    for start, end in ranges_to_remove:
        if i == start:
            skip_until = end
            in_removal_range = True
            break
    
    if not in_removal_range:
        new_lines.append(line)

# Write the cleaned file
with open('core/technical_analysis.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"\n✅ Successfully removed {len(ranges_to_remove)} unused functions")
print(f"📊 Original: {len(lines)} lines → Cleaned: {len(new_lines)} lines")
print(f"🗑️  Removed: {len(lines) - len(new_lines)} lines ({((len(lines) - len(new_lines)) / len(lines) * 100):.1f}%)")
