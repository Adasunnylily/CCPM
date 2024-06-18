import os
import json
import openai
import numpy as np
import data_utils
import time
import datetime
import copy

 
classes = np.load('cls_names.npy', allow_pickle=True)
classes = tuple(classes)
 
print("Class List:")
print(classes)

from openai import OpenAI
client = OpenAI(api_key='xxx')

#build ICL format
#in-context samples for oxfort pet 
example = 'List the most visual important features for recognizing the main object in the image as a \"Bengal\":\n\n-a long, thick tail\n-a short, blunt muzzle\n-large ears\n-large paws with sharp claws\n-patchy orange and black fur\n\nList the most visual important features for recognizing the main object in the image as a \"shiba inu\":\n\n-a black mask on the face\n-a bushy tail\n-a reddish-brown coat\n-dark, almond-shaped eyes\n-erect, triangular ears\n\n'


#query = 'List the most visual important features for recognizing the main object in the image as a \"{}\":'
query1 = 'List the most visual important features for recognizing something as a \"{}\":'

#instruction 
prompt1 =  'Now you are an assistant to help me design a design a concept set given a class name.' \
                'Concretely, a concept set denotes a set of "important visual attributes for recognizing".' \
                'These visual attributes as concepts should be relevant to the class and helpful for recognizing the main object in the image as the class.' \
                'These attributes should be descriptive and should not repeat each other.' \
                'Next, I will give you several examples for you to understand this task.' \
                f'\n{example}'
 
feature_dict = {}
for i, label in enumerate(classes):
    feature_dict[label] = set()
    print("\n", i, label)
    for _ in range(3):
        try:
            print(query.format(label))
            #client.chat.completions.create
            response = client.completions.create(
            #response = client.chat.completions.create (
                model="gpt-3.5-turbo-instruct", 
                #model='gpt-4',
                # messages=[
                #             # {"role": "system", "content": "You are a helpful assistant."},
                #             # {"role": "user", "content": "Who won the world series in 2020?"},
                #             # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                #             # {"role": "user", "content": "Where was it played?"},
                #             # base_prompt.format(label)
                #             {"role": "system", "content": "You are a helpful felinologist or cynologist."},
                #             {"role": "user", "content": "List the most important visual features for recognizing something as a \"goldfish\":"},
                #             {"role": "assistant", "content": "\n\nbright orange color\n-a small, round body\n-a long, flowing tail\n-a small mouth\n-orange fins\n\n"},
                #             {"role": "user", "content": "List the most important visual features for recognizing something as a \"beerglass\":"},
                #             {"role": "assistant", "content": "n\na tall, cylindrical shape\n-clear or translucent color\n-opening at the top\n-a sturdy base\n-a handle\n\n"},
                #             {"role": "user", "content": base_prompt.format(label)}
                #         ],
                prompt = prompt1 + query1.format(label),
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
                )
             
            #clean up responses
            #features = response["choices"][0]["text"]
            features = response.choices[0].text
            #features = response.choices[0].message.content
            features = features.split("\n-")
            features = [feat.replace("\n", "") for feat in features]
            features = [feat.strip() for feat in features]
            features = [feat for feat in features if len(feat)>0]
            features = set(features)
            feature_dict[label].update(features)
            
            #ans = response['choices'][0]['message']['content']
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
           
            feature_dict_save = copy.deepcopy(feature_dict)     
            for key, value in feature_dict_save.items():
                feature_dict_save[key] = convert_sets_to_lists(value)
    
            if try_path1 is not None:
                with open(try_path1, 'w') as f:
                    f.write(f"\n----------{now}----------\n")
                    json.dump(feature_dict_save, f, indent=3,default=str)
            # time.sleep(17)
            #return ans
        
        except Exception as e:
            print('[ERROR]', e)
            ans =  '#ERROR#'
            time.sleep(20)
            if err_file is not None:
                now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(err_file, 'a') as f:
                    f.write(f"\n----------{now}----------\n[INPUT]{base_prompt.format(label)}\n[ERROR]{e}\n")
                
            
    feature_dict[label] = sorted(list(feature_dict[label]))
    # 追加结果到同一文件
    with open(try_path, 'a') as w:
        # 添加一个分隔符，以便区分每个类别的结果
        w.write("\n")
        json.dump(feature_dict, w, indent=3)

json_object = json.dumps(feature_dict, indent=4)
with open("{}.json".format(dataset), "w") as outfile:
    outfile.write(json_object)