import joblib
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('C:\\Users\\kaiov\\Desktop\\Disease-Chatbot\\data\\processed\\Training.csv')
training = training.drop(['index'], axis=1)
cols = training.columns[1:]
x = training[cols]
y = training['label']


reduced_data = training.groupby(training['label']).max()

#Categoziração de strings
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

clf = joblib.load('C:\\Users\\kaiov\\Desktop\\Disease-Chatbot\\model\\finalized_model.sav')

scores = cross_val_score(clf, x_test, y_test)

print(scores.mean())

features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

def getSeverityDict():
    global severityDictionary
    with open('C:\\Users\\kaiov\\Desktop\\Disease-Chatbot\\data\\symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                item = {row[0]:int(row[1])}
                severityDictionary.update(item)
        except:
            pass

def calc_condition(exp,days):
    sum = 0
    for item in exp:
         sum = sum + severityDictionary[item]
    if((sum * days) / (len(exp) + 1) > 13):
        print("---------------------------------")
        print("Você deveria consultar um médico.")
    else:
        print("---------------------------------------------------------")
        print("Não parece grave, mas você deve realizar alguns cuidados.")

def getDescription():
    global description_list
    with open('C:\\Users\\kaiov\\Desktop\\Disease-Chatbot\\data\\symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            descricao = {row[0]:row[1]}
            description_list.update(descricao)

def getprecautionDict():
    global precautionDictionary
    with open('C:\\Users\\kaiov\\Desktop\\Disease-Chatbot\\data\\symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            precaucao={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(precaucao)

def getInfo():
    # name=input("Name:")
    print("Qual é seu nome? \n\t\t\t\t\t\t",end="->")
    name = input("")
    print(f"Olá, ", name)

def check_pattern(dis_list,inp):
    import re
    pred_list = []
    ptr = 0
    patt = "^" + inp + "$"
    regexp = re.compile(inp)
    for item in dis_list:
        # print(f"comparing {inp} to {item}")
        if regexp.search(item):
            pred_list.append(item)
            # return 1,itemkai
    if(len(pred_list) > 0):
        return 1, pred_list
    else:
        return ptr, item
        
def sec_predict(symptoms_exp):
    df = pd.read_csv('C:\\Users\\kaiov\\Desktop\\Disease-Chatbot\\data\\processed\\Training.csv')
    df = df.drop(['index'], axis=1)
    X = df.iloc[:,1:]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {}

    for index, symptom in enumerate(X):
        symptoms_dict[symptom] = index

    input_vector = np.zeros(len(symptoms_dict))

    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1


    return rf_clf.predict([input_vector])

def print_disease(node):
    #print(node)
    node = node[0]
    #print(len(node))
    val  = node.nonzero() 
    # print(val)
    disease = le.inverse_transform(val[0])

    return disease

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    # print(tree_)
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []

    # conf_inp=int()
    while True:

        print("Qual sintoma você está sentindo?  \n\t\t\t\t\t\t",end="->")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf==1:
            print("Buscas relacionadas ao sintoma informado: ")
            for num, it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Selecione o que mais se aproxima ao que você está sentindo (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0
            disease_input=cnf_dis[conf_inp]
            break
            # print("Did you mean: ",cnf_dis,"?(yes/no) :",end="")
            # conf_inp = input("")
            # if(conf_inp=="yes"):
            #     break
        else:
            print("Entre com um sintoma válido.")

    while True:
        try:
            num_days = int(input("Ok. Por quantos dias está sentindo isto? "))
            break
        except:
            print("Entre com o número de dias.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if  val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            # print( "You may have " +  present_disease )
            red_cols = reduced_data.columns 
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            # dis_list=list(symptoms_present)
            # if len(dis_list)!=0:
            #     print("symptoms present  " + str(list(symptoms_present)))
            # print("symptoms given "  +  str(list(symptoms_given)) )
            print("Além disto. Você também está sentindo:")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms.replace('_',' '),"? : ",end='')
                while True:
                    inp = input("")
                    if(inp == "sim" or inp=="nao"):
                        break
                    else:
                        print("Digite somente (sim ou nao) : ",end="")
                if(inp == "sim"):
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            # print(second_prediction)
            calc_condition(symptoms_exp, num_days)
            if(present_disease[0] == second_prediction[0]):
                print(f"Você pode estar com ", present_disease[0])
                print("-----------DESCRIÇÃO-------------")
                print(description_list[present_disease[0]])

                # readn(f"You may have {present_disease[0]}")
                # readn(f"{description_list[present_disease[0]]}")

            else:
                print(f"Você pode estar com ", present_disease[0], "OU com ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            # print(description_list[present_disease[0]])
            precution_list = precautionDictionary[present_disease[0]]
            print("--------------------------")
            print("Tome as seguintes precauções : ")
            for  i, j in enumerate(precution_list):
                print(i+1,")",j)

            #print("--------------------------")
            #print("Nível de confiança de predição : ")
            #confidence_level = (1.0*len(symptoms_present)) / len(symptoms_given)
            #print("Confianção é de:  " + str(confidence_level))

    recurse(0, 1)
getSeverityDict()
getDescription()
getprecautionDict()
getInfo()
tree_to_code(clf,cols)