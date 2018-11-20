import numpy as np
import dill as pickle
import python_basics
#import python_basics_soln

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def get_answers(module, fun_name, test_cases):
    f = getattr(module, fun_name)
    toR = []
    for test_case in test_cases:
        toR.append(f(*test_case))
    return toR

def get_thresh_answers(thresh, test_cases):
    t = python_basics.threshold_factory(thresh)
    toR = []
    for test_case in test_cases:
        toR.append(t(*test_cases))
    return toR

def grade(soln, submission, compares):
    toR = {}
    for fn in soln.keys():
        compare = compares[fn]
        diffs = []
        incorrects = []
        for sub_case, soln_case in zip(submission[fn], soln[fn]):
            assert sub_case[0] == soln_case[0], "Test cases not equal"
            correct = compare(sub_case[1], soln_case[1])
            diffs.append(correct)
            if not correct:
                incorrects.append((fn, sub_case[0], sub_case[1], soln_case[1]))
        toR[fn] = {"Test Case Results": diffs, "Errors": incorrects}
    return toR
    

def print_result(res):
    for fn in res.keys():
        print(f"Function: {fn}")
        errors = res[fn]["Errors"]
        if len(errors) == 0:
            print("\t All test cases passed!")
        else:
            for fn, args, sub_ans, soln_ans  in errors:
                if fn == "threshold_factory":
                    print("\tError")
                    print(f"\tFunction: {fn}")
                    print(f"\tArguments: 10, followed by {args}")
                    print(f"\tStudent Result:\n----------\n{sub_ans}\n----------\n")
                    print(f"\tCorrect Result:\n----------\n{soln_ans}\n----------\n")
                else:
                    print("\tError")
                    print(f"\tFunction: {fn}")
                    print(f"\tArguments: {args}")
                    print(f"\tStudent Result:\n----------\n{sub_ans}\n----------\n")
                    print(f"\tCorrect Result:\n----------\n{soln_ans}\n----------\n")


class SomeClass:
    def __init__(self):
        pass
    def some_method(self):
        pass
    
if __name__ == "__main__":
    assert 1/2 == .5, "Use Python3, not Python2"

    make_soln = False#True
    if make_soln:
        module = python_basics_soln
    else:
        module = python_basics

    d = {}
    fun_name = "find_sum"
    test_cases = [(1, 100), (1, 10), (-10, 10)]
    answers = get_answers(module, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    fun_name = "get_info"
    test_cases = [("Scott", "old", "Redacted"), ("Scott again", 110, 0.0)]
    answers = get_answers(module, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    fun_name = "get_method_and_var_names"
    test_cases = [("a string",), (1,)]
    answers = get_answers(module, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    fun_name = "evaluate"
    test_cases = [(len, [1,2,3]), (sorted, [3, 2, 1, 4])]
    answers = get_answers(module, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    '''
    fun_name = "threshold_factory"
    test_cases = [(4,)]
    answers = get_answers(module, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))
    '''
    
    fun_name = "threshold_factory"
    test_cases = [(1,), (4,), (5,), (10,), (15,)]
    if make_soln:
        t = lambda x: x < 10
    else:
        t = module.threshold_factory(10)
        
    answers = [t(*args) if callable(t) else None for args in test_cases]


    d[fun_name] = list(zip(test_cases, answers))
    
    for k in d.keys():
        print(list(d[k]))
   
    if make_soln:
        pickle.dump(d, open("soln.pkl", "wb"))
    else:
        soln = pickle.load(open("soln.pkl", "rb"))
        compares = {fn: (lambda x, y: x == y) for fn in soln.keys()}
        #compares["threshold_factory"] = lambda f1, f2: np.all([False if not callable(f1) and callable(f2) else f1(i) == f2(i) for i in range(-100, 100)])
        res = grade(soln, d, compares)
        
        print_result(res)
