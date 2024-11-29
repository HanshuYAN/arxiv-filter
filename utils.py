# encoding: utf-8
from bs4 import BeautifulSoup as bs
import urllib.request
import pandas as pd
import os
import re
import datetime
import pytz
import tqdm
import transformers
import torch

def check_date_and_get_fname(date):
    ###->>> The date should be [new, recent, or, yyyy-mm]
    if date == "new":
        today = datetime.date.fromtimestamp(datetime.datetime.now(tz=pytz.timezone("America/New_York")).timestamp()).strftime("%Y-%m-%d")
        fname = f"{today}_new.xlsx" 
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

def extract_subject(s):
    return re.findall(r'\((.*?)\)', s)

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
        paper['subjects'] = extract_subject(dd_list[i].find("div", {"class": "list-subjects"}).text)
        paper['main_page'] = ARXIV_BASE + dt_list[i].text.strip().split("arXiv:")[-1].split("\n")[0]
        paper['title'] = dd_list[i].find("div", {"class": "list-title mathjax"}).text.replace("Title:", "").replace("\n", "").strip()
        paper['authors'] = dd_list[i].find("div", {"class": "list-authors"}).text.replace("Authors:", "").replace("\n", "").strip()
        if timestamp == "new":
            paper['abstract'] = dd_list[i].find("p", {"class": "mathjax"}).text.replace("\n", "").strip()
        else:
            paper['abstract'] = get_abstract(paper['main_page'])
        new_paper_list.append(paper)
    return new_paper_list
