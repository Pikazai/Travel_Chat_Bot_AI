"""
Date utility functions.
"""

from datetime import datetime
from typing import Optional, Tuple


def parse_date_range(start_str: Optional[str], end_str: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Parse date strings to datetime objects.
    
    Args:
        start_str: Start date string (YYYY-MM-DD)
        end_str: End date string (YYYY-MM-DD)
        
    Returns:
        Tuple of (start_date, end_date) or (None, None)
    """
    def _parse(d: Optional[str]) -> Optional[datetime]:
        if not d or d.lower() == 'null' or d == '':
            return None
        try:
            return datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            return None
    
    start_dt = _parse(start_str)
    end_dt = _parse(end_str)
    
    # Validate dates (not in past)
    today = datetime.now().date()
    if start_dt and start_dt.date() < today:
        start_dt = None
    if end_dt and end_dt.date() < today:
        end_dt = None
    
    if start_dt and not end_dt:
        end_dt = start_dt
    
    return start_dt, end_dt

