import sys
import subprocess


PROJECT_ROOT = '~/medical-statuses/'
GOLDEN_STANDARD = PROJECT_ROOT + 'GoldenStandard.txt'

import logging
logging.basicConfig(level=logging.ERROR)


def readTrainingData():
    # Open the workbook
    xl_workbook = xlrd.open_workbook(PROJECT_ROOT+'Bigrams.xlsx')
    xl_sheet = xl_workbook.sheet_by_index(1)
    num_cols = xl_sheet.ncols   # Number of columns
    print ('Sheet name: %s' % xl_sheet.name)
    sentences1 = []
    sentences2 = []

#     for row_idx in range(0, xl_sheet.nrows):
    for row_idx in range(0, xl_sheet.nrows): 
        for col_idx in range(0, num_cols):  # Iterate through columns
            cell_obj = xl_sheet.cell(row_idx, col_idx)  # Get cell object by row, col
            if col_idx == 0:
                sentences1.append(str(cell_obj.value))
            if col_idx == 1:
                sentences2.append(str(cell_obj.value))
    return [sentences1, sentences2]



dataExcel = readTrainingData()
len(dataExcel[0])


 import spacy
 nlp = spacy.load('bg')



def removeAllPunctuations(g):
    g= [sub.replace("."," ") for sub in g]
    g= [sub.replace(","," ") for sub in g] 
    g= [sub.replace("'","") for sub in g]
    g= [sub.replace("-"," ") for sub in g]
    g= [sub.replace("/"," ") for sub in g]
    g= [sub.replace(":"," ") for sub in g]
    g= [sub.replace(";"," ") for sub in g]
    g= [sub.replace('"',"") for sub in g]
    g= [sub.replace("*","") for sub in g]
    g= [sub.replace("?"," ") for sub in g]
    g= [sub.replace("&","and") for sub in g]
    g= [sub.replace("+"," ") for sub in g]
    g= [sub.replace("["," ") for sub in g]
    g= [sub.replace("]"," ") for sub in g]
    g= [sub.replace("("," ") for sub in g]
    g= [sub.replace(")"," ") for sub in g]
    g= [sub.replace("<"," ") for sub in g]
    g= [sub.replace(">"," ") for sub in g]
    g= [sub.replace("="," ") for sub in g]
    g= [sub.replace(","," ") for sub in g]
#     g= re.sub( '\s+', ' ', g ).strip()
    return g
        
def getAverageVectorForData(data):
    data1 = removeAllPunctuations(data[0])
    data2 = removeAllPunctuations(data[1])
    return [data1, data2]



data = getAverageVectorForData(dataExcel)
len(data[0])
len(data[1])


 import numpy as np
 from sklearn.linear_model import LogisticRegression


 def trainingLogisticRegression(data):
     ## Training takes as input getAverageVectorForData(data)
     predictions = []
     evaluations = []

     for x in range(len(data[0])):
         s1 = data[0][x]
         s2 = data[1][x]

         v12 = nlp(s1 + s2).vector
         v21 = nlp(s2 + s1).vector

         predictions.append(v12)
         evaluations.append(1)
         predictions.append(v21)
         evaluations.append(10)

     # print("evaluation before", evaluations)
     predictions = np.array(predictions)
     evaluations = np.array(evaluations)

     regression = LogisticRegression()
     classifier = regression.fit(predictions,evaluations)
     print("Accuracy on training set",classifier.score(predictions,evaluations))
     return classifier


 classifierLog = trainingLogisticRegression(data)


import sys
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")



import numpy as np
# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[13]:


import tensorflow as tf



def trainingBert(data, bert_model):    
    from transformers import BertTokenizer, BertForNextSentencePrediction
    import torch

    model = BertForNextSentencePrediction.from_pretrained(bert_model, return_dict=True)
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    sentence1 = data[0]
    sentence2 = data[1]


    max_len = 1500



    input_ids = []
    attention_masks = []
    labels = []


    for x in range(len(data[0])):
        s1 = data[0][x]
        s2 = data[1][x]


        encoded_dict = tokenizer.encode_plus(
                        s1, text_pair=s2,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        truncation=True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )

        encoded_dict_reverse = tokenizer.encode_plus(
                        s2, text_pair=s1,
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length =max_len,           # Pad & truncate all sentences.
                        truncation=True,
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(0)


    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)






    from torch.utils.data import TensorDataset, random_split

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    batch_size = 16

    train_dataloader = DataLoader(
                train_dataset,  # The training samples.
                batch_size = batch_size # Trains with this batch size.
            )
    validation_dataloader = DataLoader(
                val_dataset, # The validation samples.
                batch_size = batch_size # Evaluate with this batch size.
            )
    from transformers.optimization import AdamW
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    from transformers import get_linear_schedule_with_warmup

    epochs = 4


    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer)



    training_stats = []

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
       
        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            res1 = model(b_input_ids, 
                                 token_type_ids=None, 
                                 attention_mask=b_input_mask, 
                                 next_sentence_label=b_labels)
            loss = res1[0]
            logits = res1[1]

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)            

        import os
        model_dir = str(epoch_i) + '/'
        output_dir = PROJECT_ROOT + 'model_save/' + model_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))

        print("")
        print("Running Validation...")
        # Tracking variables 
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0]
            b_input_mask = batch[1]
            b_labels = batch[2]

            res2 = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   next_sentence_label=b_labels)
            loss = res2[0]
            logits = res2[1]

            total_eval_loss += loss.item()
            logits = logits.numpy()
            label_ids = b_labels.numpy()

            accs = calc_acc(logits, label_ids)
            total_eval_accuracy += accs

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))


        print("")
        print("Training complete!")

        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
    
    
    return model
        



BERT_MODEL = 'bert-base-multilingual-cased'
bert = trainingBert(data, BERT_MODEL)




def getProbabilitiesBert(a, allSentencesInNextSystem, model):
    max_len = 1500


    input_ids = []
    attention_masks = []
    sentences = []
    labels = []

    
    for x in range(len(allSentencesInNextSystem)):
        s1 = a
        s2 = allSentencesInNextSystem[x]

        encoded_dict = tokenizer.encode_plus(
                        s1+s2, # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = max_len,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        truncation=True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        sentences.append(s1+s2)
        
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        labels.append(0)


    # Set the batch size.  
    batch_size = 16


    print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

    predictions , true_labels = [], []

    for batch in prediction_dataloader:
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids, token_type_ids=None, 
                          attention_mask=b_input_mask)

        logits = outputs[0]

        predictions.append(logits)
        true_labels.append(label_ids)

    print('    DONE.')


    result = []
    for i in range(len(allSentencesInNextSystem)):
        #if classes are 0 and 1
        result.append(
            [
                predictions[i][0],
                allSentencesInNextSystem[i]
            ]
        )
    return result




def biggest(probabilities):
    current = probabilities[0]
    for i in probabilities:
        if(current[0] < i[0]):
            current = i
    return current



 def getProbabilities(a, allSentencesInNextSystem, model, tokenizer):
     testData = []
     for b in allSentencesInNextSystem:
        ab = nlp(a + b).vector
         testData.append(ab)
     npTestData = np.array(testData)
     proba = model.predict_proba(npTestData)


     result = []
     for i in range(len(allSentencesInNextSystem)):
         #if calsses are 0 and 1
         result.append([proba[i][1], allSentencesInNextSystem[i]])
     return result
    
    
def getSystemOfSenetence(sentence, systemToSentences):
    for x in systemToSentences:
        if sentence in systemToSentences[x]:
            return x

        
def prepareSystemsForTheCurrentStatus(): 
    sentenceCount = random.randrange(4, 8)
    optionalSentenceCount = sentenceCount - 4
    choosenOptionalSentences = 0
    choosenSystemsMap = {"GeneralStatus": 1, "BodyRegions": 0,"RespiratorySystem": 1, "CardiovascularSystem":1, 'DigestiveSystem':1, 'UrogenitalSystem':0, 'EndocrineSystem':0, 'NervousSystem':0, 'SenseOrgans':0, 'MusculoskeletalSystem':0}
    optionalSystems = ['BodyRegions', 'UrogenitalSystem', 'EndocrineSystem', 'NervousSystem', 'SenseOrgans', 'MusculoskeletalSystem']
    while(choosenOptionalSentences < optionalSentenceCount):
        system = random.choices(optionalSystems, weights=(50, 30, 30, 30, 30, 50), k=1)[0]
        if(choosenSystemsMap[system] == 0):
            choosenSystemsMap[system] = 1
            choosenOptionalSentences = choosenOptionalSentences + 1
        else:
            continue
    return [choosenSystemsMap, sentenceCount]







def initializeSystemSentences(systems):
	import os.path
	print(PROJECT_ROOT + 'SystemSentences/' + 'GeneralStatus' + '.txt')
	print(os.path.exists(PROJECT_ROOT + 'SystemSentences/' + 'GeneralStatus' + '.txt'))
	assert os.path.exists(PROJECT_ROOT + 'SystemSentences/' + 'GeneralStatus' + '.txt')

	systemToSentences = {}
	for x in systems:
	    file = open(PROJECT_ROOT + 'SystemSentences/' + x + '.txt', encoding='windows-1251')
	    systemToSentences [x] = file.read().splitlines()
	return systemToSentences






def generate(currentStatus, sentenceCount, statuses, systemToSentences, systems, choosenSystemsMap, getProbabilitiesFunction, model, tokenizer):
    if(len(currentStatus) >= sentenceCount):
        statuses.append(copy.deepcopy(currentStatus))
        print(currentStatus)
        return
    
    s = currentStatus[-1]
    systemS = getSystemOfSenetence(s, systemToSentences)
    
    nextSystem = ''
    if(systems.index(systemS) == len(systems) - 1) :
        nextSystem = systems[0]
    else:
        indexOfCurrentSystem = systems.index(systemS)
        indexOfNextSystem = indexOfCurrentSystem + 1
        print(indexOfNextSystem)
        print(choosenSystemsMap)
        while indexOfNextSystem < len(choosenSystemsMap):
            if choosenSystemsMap[systems[indexOfNextSystem]] == 1:
                nextSystem = systems[indexOfNextSystem]
                break
            indexOfNextSystem = indexOfNextSystem + 1
            print(indexOfNextSystem)
            
    
    allSentencesInNextSystem = systemToSentences[nextSystem]
    probabilities = getProbabilitiesFunction(s, allSentencesInNextSystem, model, tokenizer)
    biggestProbability = biggest(probabilities)

    if(biggestProbability[0] < 0.9):
        currentStatus.append(biggestProbability[1])
        generate(currentStatus, sentenceCount, statuses, systemToSentences, systems, choosenSystemsMap, getProbabilitiesFunction, model, tokenizer)
        currentStatus.pop()
    else:
        for i in probabilities:
            if(i[0] > 0.9):
                currentStatus.append(i[1])
                generate(currentStatus, sentenceCount, statuses, systemToSentences, systems, choosenSystemsMap, getProbabilitiesFunction, model, tokenizer)
                currentStatus.pop()


systems = ['GeneralStatus', 'BodyRegions', 'RespiratorySystem', 'CardiovascularSystem', 'DigestiveSystem', 'UrogenitalSystem', 'EndocrineSystem', 'NervousSystem', 'SenseOrgans', 'MusculoskeletalSystem']
systemToSentences = initializeSystemSentences(systems)
for x in systemToSentences:
    print(x + " " + str(len(systemToSentences[x])))


def initializeGolden():
	import os.path
	print(GOLDEN_STANDARD)
	print(os.path.exists(GOLDEN_STANDARD))
	assert os.path.exists(GOLDEN_STANDARD)

	file = open(GOLDEN_STANDARD)#, encoding='windows-1251')
	golden = file.read().splitlines()
	return golden

goldenStandard = initializeGolden()
print(len(goldenStandard))
print(goldenStandard[3])


NUMBER_OF_STATUSES_TO_GENERATE = 10000
currentStatus = []
statuses = []

bert = model

 for x in range(5):
     prepSystems = prepareSystemsForTheCurrentStatus()
     choosenSystemsMap = prepSystems[0]
     sentenceCount = prepSystems[1]
     generalStatusIndex = random.randrange(1, 150)
     currentStatus.append(systemToSentences['GeneralStatus'][generalStatusIndex])
     generate(currentStatus, sentenceCount, statuses, systemToSentences, systems, choosenSystemsMap,  getProbabilities, classifierLog, '') #getProbabilitiesBert
     currentStatus.pop()



def evaluation(statuses, standard, metrics):
    stats = []
    
    for currentStatus in statuses:
        currentEvaluation = {}
        currentEvaluation['status'] = currentStatus
        for metricName in metrics:
            currentEvaluation[metricName] = metrics[metricName](standard, currentStatus)
        stats.append(currentEvaluation)
    
    df_stats = pd.DataFrame(data=stats)
    return df_stats


from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
metrics = {
    'gleu_score': sentence_gleu,
    'bleu_score': sentence_bleu,
    'meteor_score': meteor_score,
    'nist_score': sentence_nist,
}
df = evaluation(statuses, goldenStandard, metrics)

