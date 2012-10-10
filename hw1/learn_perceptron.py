import numpy as NP
import scipy.io

def learn_perceptron(neg_examples_nobias, pos_examples_nobias,
                     w_init, w_gen_feas):
    neg_examples = add_bias(neg_examples_nobias)
    pos_examples = add_bias(pos_examples_nobias) 

    D = neg_examples.shape[1]
    
    if(w_init == None):
        w = NP.random.randn(D,1)
    else:
        w = w_init
    
    iteration = 0

    (neg_err, pos_err) = eval_perceptron(neg_examples, pos_examples, w)
    num_errs = len(neg_err) + len(pos_err)
    err_history = [num_errs]
    print error_report(iteration, num_errs, w, w_gen_feas)

    while(num_errs > 0 and iteration < 1000):
        iteration += 1
        w = update_weights(neg_examples, pos_examples, w)

        (neg_err, pos_err) = eval_perceptron(neg_examples, 
                                             pos_examples, w)
        num_errs = len(neg_err) + len(pos_err)
        err_history.append(num_errs)
        print error_report(iteration, num_errs, w, w_gen_feas)
    return w        

    
def error_report(i, num_errs, w, w_gen_feas): 
    errstr = 'Number of errors in iteration %d:\t%d\n' % (i, num_errs)
    weightstr = '\tweights=%s\n' % str(w)
    if(len(w_gen_feas) == 0):
        feasiblestr = ''
    else:
        dist = NP.linalg.norm(w - w_gen_feas)
        feasiblestr = '\tdistance to feasible=%.3f' % dist
    return errstr + weightstr + feasiblestr
    
def add_bias(examples):
    return NP.hstack((examples, NP.ones((examples.shape[0],1))))

def eval_perceptron(neg_examples, pos_examples, w):
    (neg_mistakes, pos_mistakes) = ([], [])
    for (negidx, neg) in enumerate(neg_examples):
        if(NP.dot(neg, w) >= 0):
            # Wrong!
            neg_mistakes.append(negidx)
    for (posidx, pos) in enumerate(pos_examples):
        if(NP.dot(pos, w) < 0):
            # Wrong!
            pos_mistakes.append(posidx)
    return (neg_mistakes, pos_mistakes)

def update_weights(neg_examples, pos_examples, w_current):
    w = w_current.copy()
    for (negidx, neg) in enumerate(neg_examples):
        activation = NP.dot(neg, w)
        if(activation >= 0):
            # YOUR CODE HERE
    for (posidx, pos) in enumerate(pos_examples):
        activation = NP.dot(pos, w)
        if(activation < 0):
            # YOUR CODE HERE
    return w


datafiles = ['dataset1.mat', 'dataset2.mat', 
             'dataset3.mat', 'dataset4.mat']
for df in datafiles: 
    data = scipy.io.loadmat(df)

    neg_examples_nobias = data['neg_examples_nobias']
    pos_examples_nobias = data['pos_examples_nobias']
    w_init = data['w_init']
    w_gen_feas = data['w_gen_feas']

    learn_perceptron(neg_examples_nobias, pos_examples_nobias,
                     w_init, w_gen_feas)
