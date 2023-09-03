from simpletransformers.t5 import T5Model, T5Args


user_input='''
An essay is, generally, a piece of writing that gives the author's own argument, but the definition is vague, overlapping with those of a letter, a paper, an article, a pamphlet, and a short story. Essays have traditionally been sub-classified as formal and informal. Formal essays are characterized by "serious purpose, dignity, logical organization, length," whereas the informal essay is characterized by "the personal element (self-revelation, individual tastes and experiences, confidential manner), humor, graceful style, rambling structure, unconventionality or novelty of theme," etc.
'''

pred_params = {
        'max_seq_length': 512,
        'use_multiprocessed_decoding': False
        }

input_list=[]
input_text=user_input.split('.')
str=""
count=0
for sentence in input_text:
    str=str+sentence
    count=count+len(sentence)
    if(count>512):
        str=str+" "
        input_list.append(str)
        count=0
        str=""

print(len(user_input))
print(len(input_list))
for i in input_list:
    print(len(i))

# mo = T5Model('t5', 'outputs/best_model', args=pred_params, use_cuda=False)
# pred = mo.predict(input_list)
# predicted_output=''
# for i in range(len(pred)):
#     predicted_output+=pred[i]
# print("predicted output is : ",predicted_output)


# features_list=[['char_count', 'word_count', 'sent_count', 'avg_word_len', 'lemma_count', 'spell_err_count', 'noun_count', 'adj_count', 'verb_count', 'adv_count'], [2810.0, 577.0, 29.0, 4.760831889081456, 299.0, 39.0, 145.0, 54.0, 94.0, 50.0]]


# f_list={}
# for i in range(len(features_list[0])):
#     f_list[features_list[0][i]]=features_list[1][i]

# print(f_list)

#             features=fs1.drop(columns=['essay','essay_set'])
#             features_list=[features.columns.values.tolist()] + features.values.tolist()
#             print(features_list)
#             f_list=[]
#             for i in range(len(features_list[0])):
#                 f_list.append({features_list[0][i]:features_list[1][i]})

#             print(f_list)



#             {% if features %}
#                     <p>Grade : {{features}}</p>
                    
#                 {% else %}
#                     <p>No features Available..!!</p>
#                 {% endif %}

#                 <p>Char count : {{features[char_count}}</p>
#                 <p>Word count : {{features[word_count}}</p>
#                 <p>Sentence count : {{features[sent_count}}</p>
#                 <p>Spell errors: {{features[spell_err_count}}</p>
#                 <p>Noun count : {{features[noun_count}}</p>
#                 <p>Adjective count : {{features[adj_count}}</p>
#                 <p>Verb count : {{features[verb_count}}</p>
#                 <p>Adverb count : {{features[adv_count}}</p>

#                 {% for key, value in dict_item.items() %}
#                     Key: {{key}} Value: {{value}}
#                 {% endfor %}