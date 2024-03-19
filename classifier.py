import csv

def posterior(prior, likelihood, observation):
    prob_class_is_true = prior
    prob_class_is_false = (1 - prior)
    for i in range(0, len(observation)):
        if observation[i] == True:
            prob_class_is_true *= likelihood[i][True]
            prob_class_is_false *= likelihood[i][False]
        else:
            prob_class_is_true *= (1 - (likelihood[i][True]))
            prob_class_is_false *= (1 - (likelihood[i][False]))

    return prob_class_is_true / (prob_class_is_true + prob_class_is_false)

def learn_prior(file_name, pseudo_count=0):
    with open(file_name) as in_file:
            training_examples = [tuple(row) for row in csv.reader(in_file)]
    
    data_length = len(training_examples[1:])
    features = [i for i in training_examples[0]]
    data = {i: 0 for i in features} # {'X1': 0, 'X2': 0, 'X3': 0, 'X4': 0, 'X5': 0, 'X6': 0, 'X7': 0, 'X8': 0, 'X9': 0, 'X10': 0, 'X11': 0, 'X12': 0, 'SPAM': 0}

    for row in range(1, len(training_examples)):
        for i in range(0, len(features)):
            data[features[i]] += int(training_examples[row][i])

    prior = (data['SPAM'] + pseudo_count) / (data_length + (pseudo_count * 2))
    return prior


def learn_likelihood(file_name, pseudo_count=0):
    # likelihood[i][False] is P(X[i]=true|Spam=false) and likelihood[i][True] is P(X[i]=true|Spam=true)
    with open(file_name) as in_file:
            training_examples = [tuple(row) for row in csv.reader(in_file)]
    
    data_length = len(training_examples[1:])
    features = [i for i in training_examples[0]]
    data_true = {i: 0 for i in features} # {'X1': 0, 'X2': 0, 'X3': 0, 'X4': 0, 'X5': 0, 'X6': 0, 'X7': 0, 'X8': 0, 'X9': 0, 'X10': 0, 'X11': 0, 'X12': 0, 'SPAM': 0}
    data_false = {i: 0 for i in features}
    class_true_count = 0
    class_false_count = 0
    
    for row in range(1, len(training_examples)):
        if int(training_examples[row][-1]) == 1:
            class_true_count += 1
        else:
            class_false_count += 1
        for i in range(0, len(features) - 1):
            if int(training_examples[row][-1]) == 1: # spam is true
                data_true[features[i]] += int(training_examples[row][i])
            else: # spam is false
                data_false[features[i]] += int(training_examples[row][i])

    likelihoods = [(0, 0) for feature in range(0, len(features) - 1)]
    for i in range(0, len(features) - 1):
        likelihoods[i] = ((data_false[features[i]] + pseudo_count) / (class_false_count + (2 * pseudo_count)), (data_true[features[i]] + pseudo_count) / (class_true_count + (2 * pseudo_count)))

    return likelihoods


def nb_classify(prior, likelihood, input_vector):
    chance = posterior(prior, likelihood, input_vector)
    if chance <= 0.5:
        return ("Not Spam", (1 - chance))
    return ("Spam", chance)


prior = learn_prior("spam-labelled.csv")
likelihood = learn_likelihood("spam-labelled.csv")

input_vectors = [
    (1,1,0,0,1,1,0,0,0,0,0,0),
    (0,0,1,1,0,0,1,1,1,0,0,1),
    (1,1,1,1,1,0,1,0,0,0,1,1),
    (1,1,1,1,1,0,1,0,0,1,0,1),
    (0,1,0,0,0,0,1,0,1,0,0,0),
    ]

predictions = [nb_classify(prior, likelihood, vector) 
               for vector in input_vectors]

for label, certainty in predictions:
    print("Prediction: {}, Certainty: {:.5f}"
          .format(label, certainty))
