from __future__ import division, print_function
import numpy as np

# !pip install simpletransformers
# from simpletransformers.t5 import T5Model
# Flask utils
from flask import Flask,request, render_template
#ts utils
import numpy as np
import pickle
import pandas as pd
from aes_preprocess import *

pkl_filename = "testmodel.pkl"
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Define a flask app
app = Flask(__name__)



@app.route('/', methods=['GET'])
def index():
    # Main page
    print("index")
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            user_input= request.form['input_text']
            pred_params = {
                    'max_seq_length': 512,
                    'use_multiprocessed_decoding': False
                    }
            # list of sentences
            input_list=[]

            input_text=user_input.split('.') #list of sentences
            print
            str="" #para
            count=0 #length pf para
            for sentence in input_text:
                str=str+sentence
                count=count+len(sentence)
                if(count>128):
                    input_list.append(str)
                    count=0
                    str=""
            

            mo = T5Model('t5', 'outputs/best_model', args=pred_params, use_cuda=False)
            pred = mo.predict(input_list)
            predicted_output=''
            for i in range(len(pred)):
                predicted_output=predicted_output+pred[i]+"."
            print("predicted output is : ",predicted_output)
            if(len(user_input.split(' '))>300):
                disabled=False
            else:
                disabled=True
            return render_template('prediction.html',original_text=user_input,summarized_text=predicted_output,disabled=disabled)
        except Exception as e:
            print(e)
            return render_template('index.html')
    return None

@app.route('/grading', methods=['GET', 'POST'])
def grade():
    if request.method == 'POST':
        try:
            user_input= request.form['input_text']
            print(user_input)

            inp='''

            My first campaign as spokesman and strategist for Tony Blair was in 1997, three years in the planning after he had become leader of the Opposition  Labour Party. Some of the principles of strategy we applied back then would certainly apply to a modern day election. But their tactical execution almost certainly would not. Politicians and their strategists have to adapt to change as well as lead it. Seguela gives some interesting insights into those who have adapted well, and those who have done less well. He clearly adores former President Lula of Brazil and you can feel his yearning for a French leader who can somehow combine hard-headed strategy with human empathy in the same way as a man who left office with satisfaction ratings of 87percent. Seguela probably remains best known in political circles for his role advising Francois Mitterrand. Yet wheras I am ‘tribal Labour’, and could not imagine supporting a Conservative Party candidate in the UK, Seguela came out as a major supporter of Nicolas Sarkozy. I wonder if one of the reasons was not a frustration that large parts of the left in France remain eternally suspicious of modern communications techniques and styles which, frankly, no modern leader in a modern democracy can ignore. How he or she adapts to, or uses, them is up to them. But you cannot stand aside and imagine the world has not changed.

            '''
            # creating a dataframe with essay
            df=pd.DataFrame({'essay':[user_input],'essay_set':[1]})

            #  Vectorizing array 
            a,b= get_count_vectors(df['essay'])

            # matrix to numpy array
            xcv = b.toarray()

            # As model is trained with shape(600*10000) setting extra characters to zero
            new=np.pad(xcv[0], (0, (10000-len(xcv[0]))), 'constant')

            # Extracting features for the given essay for different metrics
            fs1 = extract_features(df)

            # Adding vectors and features
            x = np.concatenate((fs1.iloc[:, 1:].to_numpy(), [new]), axis = 1)
            # predicting the output using the saved model 
            predicted_grade= pickle_model.predict(x)

            predicted_grade=np.around(predicted_grade)
            features=fs1.drop(columns=['essay','essay_set'])
            features_list=[features.columns.values.tolist()] + features.values.tolist()
            print(features_list)
            f_list={}
            for i in range(len(features_list[0])):
                f_list[features_list[0][i]]=features_list[1][i]
            print(f_list)
            
            
            return render_template('grading.html',original_text=user_input,predicted_grade=min(predicted_grade[0],10),features=f_list)
        except Exception as e:
            print(e)
            return render_template('index.html')
    return None
if __name__ == '__main__':
    
    app.run(debug=True)

