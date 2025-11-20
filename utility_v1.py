#run a sampling/analysis method and save out its progress at n_timepoints equally spaced* points
# *equally spaced by wall clock time
# parameters:
#   analysis_method: the sampling and analysis method to analyze
#   params: parameters for the analysis method, including the wall clock time per time point
#   initial_state: initial state of the system, including both coordinates and other inputs such as the metadynamics potential grid
#   n_timepoints: number of serial simulation segments to run
# returns:
#   a list of the observables calculated by the method at each time point

def run_for_n_timepoints(analysis_method, params, initial_state, n_timepoints):

    method_state = initial_state
    method_output = []

    for t in range(n_timepoints):
        print(f"running segment {t+1} of {n_timepoints}", end='\r')
        method_state, observable, stopflag = analysis_method(method_state, params)
        method_output.append(observable)
        if stopflag:
            print("method halted by exceeding aggregate simulation limit")
            break

    return method_output