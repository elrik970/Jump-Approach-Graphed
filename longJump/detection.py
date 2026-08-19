# normalize(), filter_peaks() find_contact_frames()

def filterPeaksbyTimeBetween(peaks, min_distance, max_distance, consecutive_peaks):
    if len(peaks) < 2:
        return list(peaks)
        
    # --- PHASE 1: FIND WHERE THE RUNNING RHYTHM STARTS ---
    first_peak = 0
    valid_gap_counter = 0
    
    for i in range(len(peaks) - 1):
        gap = peaks[i+1] - peaks[i]
        
        if min_distance <= gap <= max_distance:
            valid_gap_counter += 1
        else:
            # If the rhythm breaks before we hit our goal, reset the counter
            valid_gap_counter = 0 
            
        # Once we find enough valid steps in a row, lock in the start index
        if valid_gap_counter >= consecutive_peaks:
            # Go back to where this continuous chain started
            first_peak = i + 1 - consecutive_peaks 
            break

    print(f"First_peak: {first_peak}")

    filteredPeaks = [peaks[first_peak]]
    n = len(peaks)
    
    for i in range(first_peak+1, n):
        
        # Check distance to the previous peak
        dist_back = peaks[i] - filteredPeaks[-1]
        if min_distance <= dist_back <= max_distance:
            filteredPeaks.append(peaks[i])
            
    return filteredPeaks

def GetApproachBounds(corr, corr_threshold_start, corr_threshold_end, corr_rolling_average_scalar, window_size, fps):
    """ Gets the start and end frame of the approach using the two ankles movement corrrelation value 
    Args:
        corr: Series representing each frames correlation value (1 means legs are moving same direction, -1 means opposite direcitons)
    """
    start_frame = 0
    end_frame = 0

    # Calculate half-window size to shift for lag (e.g., 15 frames total -> 7 frames lag)
    window_size = int(round(fps * window_size))
    half_window = round(window_size / 2)
    print(corr)

    rolling_corr = corr.rolling(window=(round(fps * corr_rolling_average_scalar)), center=True).mean()
    rolling_corr = rolling_corr.fillna(0)
    mid_sprint = rolling_corr.idxmin()

    print(mid_sprint)

    before_sprint = corr.iloc[:mid_sprint] 
    after_sprint = corr.iloc[mid_sprint:]


    start_candidates = before_sprint[before_sprint > corr_threshold_start]
    if not start_candidates.empty:
        # The last index in this filtered group is physically closest to mid_sprint
        raw_start = start_candidates.index[-1]
    else:
        raw_start = corr.index[0] # Default to start of video


    end_candidates = after_sprint[after_sprint > corr_threshold_end]
    if not end_candidates.empty:
        raw_end = end_candidates.index[0] # First index where correlation breaks down
    else:
        raw_end = corr.index[-1] # Default to end of video

    # raw_end = (after_sprint > corr_threshold_end).idxmax()

    print(raw_end, half_window)



        
    
    # 4. Shift back/forward by half the window to catch the true physical movement
    start_frame = max(0, raw_start - half_window)
    end_frame = min(len(corr) - 1, raw_end + half_window)
    
    return int(start_frame), int(end_frame), int(mid_sprint), rolling_corr

def FilterPeaksByApproachStartAndEnd(sprint_peaks, start, end):
    new_sprint_peaks = []
    for peak in sprint_peaks:
        if (int(start) <= int(peak) <= int(end)):
            new_sprint_peaks.append(peak)
    return new_sprint_peaks


class InitialContactNotFound(Exception):
    """Raised when unable to confidently detect the initial contact of the athlete after maximum extension (scipy.find_peaks found 0 peaks in vertical deacelleration graph)."""
    pass
