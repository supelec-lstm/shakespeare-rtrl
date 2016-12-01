from network import *

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',\
            'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',\
            'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ',\
            ',', '.', '?', ';', ':', "'", '"', '[', ']',\
             '-', '(', ')', '&', '!']

letter_to_index = {letter: i for i, letter in enumerate(letters)}
index_to_letter = {i: letter for i, letter in enumerate(letters)}

def string_to_sequence(string):
    sequence = np.zeros((len(string), len(letters)))
    for i, letter in enumerate(string):
        sequence[i,letter_to_index[letter]] = 1
    return sequence

def letter_to_onehot(string):
    sequence = [0 for _ in letters]
    sequence[letter_to_index[string]] = 1
    return sequence

def onehot_to_letter(onehot):
    for i in range(len(letters)):
        if onehot[0][i]:
            return index_to_letter[i]

def indice_max(l):
    m = 0
    i_m = 0
    for i, val in enumerate(l):
        if m < val:
            m = val
            i_m = i
    return i_m

def train_reber(path, N, reuse = False):
    if reuse == False:
        network = Network(len(letters), 60, len(letters))
    f = open(path)
    learning_rate = 0.1
    for i in range(N):
        if i % 1000 == 0:
             pickle.dump(network, open('network2.{}.pickle'.format(i),'wr'))
             print(i/float(N), '%')
        #print(i)
        string = f.readline().strip().upper()
        #print(string)
        network.train_sequence(string_to_sequence(string), learning_rate)
    f.close()
    return network

def predict_correctly(network, string, threshold):
    network.reset_memoization()
    for i, x in enumerate(string_to_sequence(string)[:-1]):
        y = network.propagate(x)
        expected_index = letter_to_index[string[i+1]]
        if y[expected_index] < threshold:
            return False
    return True



if __name__ == '__main__':
    for _ in range(1):
        #network = train_reber('shakespear2.txt', 50000)
        # Save the network
        #network = pickle.load(open('network49000.pickle', 'rb'))

     for k in range(1,40):
        Y = 'S'
        network = pickle.load(open('network{}000.pickle'.format(k), 'rb'))
        for i in range(1000):
	    y = network.propagate(letter_to_onehot(Y[-1]))
	    #print(y[indice_max(y)])
	    Y += index_to_letter[indice_max(y)]
	    #print(Y[-1])

        f = open("shakespeare_produced.txt".format(k),"a")
        f.write("{}000 lignes passees \n".format(k) + Y + "\n")
        f.close()
