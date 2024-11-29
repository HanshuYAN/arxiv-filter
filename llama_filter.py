# encoding: utf-8
import pandas as pd
import os
import tqdm
import transformers
import torch
from utils import check_date_and_get_fname, download_papers

######>================ Get paper ================>######
# field_abbr_list, date = ['cs.CV', 'cs.LG'], "recent"
field_abbr_list, date = ['cs.CV', 'cs.LG'], "new"
fname = check_date_and_get_fname(date)
papers = []
for field_abbr in field_abbr_list:
    papers.extend(download_papers(field_abbr, date, max=1000))


#####>================ Llama-3.1 ================>######
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
FOCUS_START, FOCUS_END = "<<<", ">>>"
ANS_POS, ANS_NEG = "***YES***", "***NOT***"
def parse_response(output):
    if ANS_POS in output:
        ans = "YES"
        reason = output.split(ANS_POS)[1]
    elif ANS_NEG in output:
        ans = "NOT"
        reason = output.split(ANS_NEG)[1]
    else:
        ans, reason = None, None
    return ans, reason
    
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    # device_map="auto",
    device='cuda',
)

######>================ filtering ================>######
interests_p0 = {
    'multimodal learning': ["vision language modeling"],
    'video generation': ["text to video generation",],
    'post-training': ["preference optimization", "reinforcement learning",],
    'generative models': ["flow", "diffusion", "VAE", "GAN", "fast or few-step generation"],
} # topic: method/task

for paper in tqdm.tqdm(papers, desc='Filtering papers: '):
    paper_title = paper["title"]
    paper_abstract = paper["abstract"]
    
    filter_prompt = "I am interested in the following topics:\n"
    for topic in interests_p0:
        filter_prompt += f"(topic: {topic}, method or task: {", ".join(interests_p0[topic])})\n"
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a research assistant of artificial intelligence, deep learning, and machine learning. "
                "The user will provide his/her interested topics. Each topic is exemplified by several related methods or tasks. "
                "Go through the title and abstract of a research paper given by the user. "
                f"Tell me whether the given paper aligns with at least one of the user's interest. "
                f"The title and abstract both start with {FOCUS_START} and end with {FOCUS_END}. "
                f"Output a response begining with {ANS_POS} OR {ANS_NEG} and explain why."
            )
        },
        {
            "role": "user", 
            "content": filter_prompt + (
                f"The paper title is: {FOCUS_START}{paper_title}{FOCUS_END}.\n"
                f"The paper abstract is: {FOCUS_START}{paper_abstract}{FOCUS_END}.\n"
            )
        },
    ]
    
    outputs = pipeline(messages, max_new_tokens=256,)
    outputs = outputs[0]["generated_text"][-1]['content']
    ans, reason = parse_response(outputs)
    paper["p0_interest"] = ans
    paper["p0_reason"] = reason

    print(">>>>>>>>>>>>>>>>>>>>>")
    print(paper_title)
    print(outputs, reason)
    print("<<<<<<<<<<<<<<<<<<<<<")


######>================ save to excel format, xlsx ================>###### 
parent_dir = os.path.dirname('./data')
if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)
df = pd.DataFrame(papers)
df.to_excel(os.path.join('./data',fname))

