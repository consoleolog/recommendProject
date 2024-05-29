from instagrapi import Client
from multiprocessing.dummy import Pool as ThreadPool
from crawler import crawler
from functools import partial

hashtag_list = ['일상','학생','직장인','여행']

settings = {

    'user-agent': ''
}
cl = Client(settings=settings)
cl.login(
    '',
    ''
)
partial_crawler = partial(crawler, cl=cl)

pool = ThreadPool(processes=4)
result = pool.map(partial_crawler, hashtag_list)
pool.close()
pool.join()

cookie = {
    'ig_did': '',
    'datr': '',
    'ig_nrcb': '',
    'mid': '',
    'ps_n': '',
    'ps_l': '',
    # ''
    'dpr': '',
    'shbid': '',
    'shbts': '',
    'ds_user_id': '',
    'fbsr_124024574287414': '',
    'wd': '',
    'csrftoken': '',
    'rur': '',
    'sessionid': ''
}

