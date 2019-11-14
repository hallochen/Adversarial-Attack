#coding=utf-8
"摘自https://www.biendata.com/forum/view_post_category/718/"
import pandas as pd
from xml.dom.minidom import parse
def generate_train_data_pair(equ_questions, not_equ_questions):
    a = [x+"\t"+y+"\t"+"0" for x in equ_questions for y in not_equ_questions]
    b = [x+"\t"+y+"\t"+"1" for x in equ_questions for y in equ_questions if x!=y]
    return a+b
def parse_train_data(xml_data):
    pair_list = []
    doc = parse(xml_data)
    collection = doc.documentElement
    for i in collection.getElementsByTagName("Questions"):
        # if i.hasAttribute("number"):
        #     print ("Questions number=", i.getAttribute("number"))
        EquivalenceQuestions = i.getElementsByTagName("EquivalenceQuestions")
        NotEquivalenceQuestions = i.getElementsByTagName("NotEquivalenceQuestions")
        equ_questions = EquivalenceQuestions[0].getElementsByTagName("question")
        not_equ_questions = NotEquivalenceQuestions[0].getElementsByTagName("question")
        equ_questions_list, not_equ_questions_list = [], []
        for q in equ_questions:
            try:
                equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        for q in not_equ_questions:
            try:
                not_equ_questions_list.append(q.childNodes[0].data.strip())
            except:
                continue
        pair = generate_train_data_pair(equ_questions_list, not_equ_questions_list)
        pair_list.extend(pair)
    print("All pair count=", len(pair_list))
    return pair_list
def write_train_data(file, pairs):
    with open(file, "w") as f:
        for pair in pairs:
            f.write(pair+"\n")
if __name__ == "__main__":
    pair_list = parse_train_data("train_set.xml")
    question1=[]
    question2=[]
    label=[]
    for pair in pair_list:
        pair=pair.split('\t')
        question1.append(pair[0])
        question2.append(pair[1])
        label.append(pair[2])
    df=pd.DataFrame()
    df['question1']=question1
    df['question2']=question2
    df['label']=label
    df.to_csv('train.csv',index=False,encoding='utf-8')

