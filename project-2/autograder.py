import numpy as np
import dill as pickle
import sys

make_soln = False #True
if make_soln:
    import learners_soln, preprocessors_soln, evaluators_soln

import learners, preprocessors, evaluators

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

def get_answers(module, fun_name, test_cases):
    f = getattr(module, fun_name)
    toR = []
    for test_case in test_cases:
        toR.append(f(*test_case))
    return toR


def grade(soln, submission, compares):
    toR = {}
    for fn in submission.keys():
        compare = compares[fn]
        diffs = []
        incorrects = []
        for sub_case, soln_case in zip(submission[fn], soln[fn]):
            #print(sub_case[0],soln_case[0])
            #assert hash(sub_case[0]) == hash(soln_case[0]), "Test cases not equal"
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
                print("\tError")
                print(f"\tFunction: {fn}")
                print(f"\tArguments: {args}")
                print(f"\tStudent Result:\n----------\n{sub_ans}\n----------\n")
                print(f"\tCorrect Result:\n----------\n{soln_ans}\n----------\n")


def Part1():
    d = {}
    #QA: MSE
    fun_name = "MSE"
    test_cases = [(np.array(range(5), dtype=float), np.array(range(5), dtype=float)*2),
                  (np.array(range(10), dtype=float), np.sin(np.array(range(10), dtype=float)))]

    answers = get_answers(mod_evaluators, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    #QB: MAD
    fun_name = "MAD"
    test_cases = [(np.array(range(5), dtype=float), np.array(range(5), dtype=float)*2- 9),
                  (np.array(range(10), dtype=float), np.sin(np.array(range(10), dtype=float)))]

    answers = get_answers(mod_evaluators, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    #QC: CV
    fun_name = "cross_validation"
    X = np.array(range(100), dtype=float).reshape(20, 5)
    y = np.array(range(20), dtype=float)
    reg = learners.ToyRegressor()
    evaler = evaluators.zero_one
    num_folds = 10

    test_cases = [(X, y, reg, evaler, num_folds)]

    X = np.array(range(100), dtype=float).reshape(20, 5)
    y = np.array(range(20), dtype=float)
    reg = learners.ToyRegressor()
    evaler = evaluators.zero_one
    num_folds = 11

    test_cases.append((X, y, reg, evaler, num_folds))

    answers = get_answers(mod_evaluators, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))
    return d



def Part2():
    d = {}
    #QA: Prepend 1's
    fun_name = "prepend_1s"
    test_cases = [(np.array(range(20), dtype=float).reshape(4, 5),)]
    answers = get_answers(mod_preprocessors, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    #QB: Poly Lift
    fun_name = "poly_lift"
    test_cases = [(np.array(range(20), dtype=float), 3),
                  (np.array(range(20), dtype=float), 0)]
    answers = get_answers(mod_preprocessors, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    #QC: Standardize
    fun_name = "standardize"
    test_cases = [(np.array(range(20), dtype=float).reshape(4, 5),)]
    answers = get_answers(mod_preprocessors, fun_name, test_cases)
    d[fun_name] = list(zip(test_cases, answers))

    return d


def Part3_QA_QB_QC(learner_name, init_params, tr_X, tr_y, te_X):
    reg_class = getattr(mod_learners, learner_name)
    reg = reg_class(**init_params)
    reg.fit(tr_X, tr_y)
    preds = reg.predict(te_X)

    return preds

def Part3():
    d = {}
    current_module = sys.modules[__name__]
    tr_X = np.eye(10, dtype=float)
    n_tr, d_tr = tr_X.shape
    tr_y = np.dot(tr_X, np.array(range(d_tr), dtype=float) + 1)
    te_X = tr_X

    test_cases = [("OLS", {}, tr_X, tr_y, te_X),
                  ("RidgeRegression", {"lamb": 1}, tr_X, tr_y, te_X),
                  ("RidgeRegression", {"lamb": 2}, tr_X, tr_y, te_X),
                  ("RidgeRegression", {"lamb": 0}, tr_X, tr_y, te_X),
                  ("DualRidgeRegression", {"lamb" : 1, "kernel" : learners.simple_kernel}, tr_X, tr_y, te_X),
                  ("DualRidgeRegression", {"lamb" : 0, "kernel" : learners.simple_kernel}, tr_X, tr_y, te_X)]
                  #("GeneralizedRidgeRegression", {"reg_weights": [i / 10 for i in range(d_tr)]}, tr_X, tr_y, te_X),
                  #("AdaptiveLinearRegression", {"kernel" : learners.simple_kernel}, tr_X, tr_y, te_X)]
    answers = get_answers(current_module, "Part3_QA_QB_QC", test_cases)

    d["Part3_QA_QB_QC"] = list(zip(test_cases, answers))
    return d


if __name__ == "__main__":
    if make_soln:
        mod_learners = learners_soln
        mod_preprocessors = preprocessors_soln
        mod_evaluators = evaluators_soln
    else:
        mod_learners = learners
        mod_preprocessors = preprocessors
        mod_evaluators = evaluators

    d = {}
    if len(sys.argv) == 1:# or make_soln:
        d = {**Part1(), **Part2(), **Part3()}
    else:
        if "p1" in sys.argv:
            d.update(Part1())
        if "p2" in sys.argv:
            d.update(Part2())
        if "p3" in sys.argv:
            d.update(Part3())


    #for k in d.keys():
    #    print(list(d[k]))


    if make_soln:
        pickle.dump(d, open("soln.pkl", "wb"))
    else:
        soln = pickle.load(open("soln.pkl", "rb"))
        compares = {fn: (lambda x, y: x is not None and y is not None and np.allclose(x,y, atol = 1e-20)) for fn in soln.keys()}
        res = grade(soln, d, compares)

        print_result(res)
    print("Finished")
    print("Note: Questions D and E of Part 3 are NOT tested by the autograder. You'll want to create your own tests.")
