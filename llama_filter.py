# encoding: utf-8
from bs4 import BeautifulSoup as bs
import urllib.request
import pandas as pd
import os
import datetime
import pytz
import tqdm
import transformers
import torch

def check_date_and_get_fname(date):
    ###->>> The date should be [new, recent, or, yyyy-mm]
    if date == "new":
        today = datetime.date.fromtimestamp(datetime.datetime.now(tz=pytz.timezone("America/New_York")).timestamp()).strftime("%Y-%m-%d")
        fname = f"{today}.xlsx" 
        return fname
    elif date == "recent":
        today = datetime.date.fromtimestamp(datetime.datetime.now(tz=pytz.timezone("America/New_York")).timestamp()).strftime("%Y-%m-%d")
        fname = f"{today}_recent.xlsx" 
    else:
        try:
            datetime.datetime.strptime(date, "%Y-%m")
        except:
            raise ValueError("The date should be [new, recent, or, yyyy-mm]")
        today = datetime.date.fromtimestamp(datetime.datetime.now(tz=pytz.timezone("America/New_York")).timestamp()).strftime("%Y-%m")
        if date <= today:
            fname = f"{date}-xx.xlsx"
            return fname
        else:
            raise ValueError("The date should not be in the future.")
    return fname


VALID_FIELD_ABBR = ['cs.LG', 'cs.CV']
ARXIV_BASE = "https://arxiv.org/abs/"

def get_abstract(paper_url):
    page = urllib.request.urlopen(paper_url)
    soup = bs(page, features="html.parser")
    content = soup.body.find("div", {'id': 'content'})
    abstract = content.find_all("blockquote", {"class": 'abstract mathjax'})[0].text
    return abstract.replace("Abstract:", "").replace("\n", "").strip()

def download_papers(field_abbr, timestamp='new', max=20):
    ###->>> Reference: https://github.com/AutoLLM/ArxivDigest/blob/main/src/download_new_papers.py
    assert field_abbr in VALID_FIELD_ABBR
    url_request = f'https://arxiv.org/list/{field_abbr}/{timestamp}?skip=0&show={max}' # the number of entries in this page is usually less than 1000
    page = urllib.request.urlopen(url_request)
    soup = bs(page, features="html.parser")
    content = soup.body.find("div", {'id': 'content'})
    dt_list = content.dl.find_all("dt")
    dd_list = content.dl.find_all("dd")
    assert len(dt_list) == len(dd_list)
    
    new_paper_list = []
    for i in tqdm.tqdm(range(len(dt_list))):
        paper = {}
        paper_number = dt_list[i].text.strip().split("arXiv:")[-1].split("\n")[0]
        paper['main_page'] = ARXIV_BASE + paper_number
        paper['title'] = dd_list[i].find("div", {"class": "list-title mathjax"}).text.replace("Title:", "").replace("\n", "").strip()
        if timestamp == "new":
            paper['abstract'] = dd_list[i].find("p", {"class": "mathjax"}).text.replace("\n", "").strip()
        else:
            paper['abstract'] = get_abstract(paper['main_page'])
        paper['authors'] = dd_list[i].find("div", {"class": "list-authors"}).text.replace("Authors:", "").replace("\n", "").strip()
        paper['subjects'] = dd_list[i].find("div", {"class": "list-subjects"}).text.split("(")[1].split(")")[0] # short
        new_paper_list.append(paper)
    return new_paper_list



######>================ Get paper ================>######
field_abbr_list, date = ['cs.CV', 'cs.LG'], "recent"
fname = check_date_and_get_fname(date)
papers = []
for field_abbr in field_abbr_list:
    papers.extend(download_papers(field_abbr, date))


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

