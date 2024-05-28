from instagrapi import Client
from multiprocessing.dummy import Pool as ThreadPool
from crawler import crawler
from functools import partial

hashtag_list = ['일상','학생','직장인','여행']

cl = Client()
cl.login(

)
partial_crawler = partial(crawler, cl=cl)

pool = ThreadPool(processes=4)
result = pool.map(partial_crawler, hashtag_list)
pool.close()
pool.join()