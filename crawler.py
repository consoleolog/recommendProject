import time
import uuid
import urllib.request
from smtp_setting import send_email
def crawler(hashtag_word, cl):
    hash_keyword = hashtag_word

    hashtag = cl.hashtag_info(hash_keyword)  # 해시태그 건수

    time.sleep(3)

    # 연관 해시태그 검색어 리스트
    # hashtag_related_hashtags = cl.hashtag_related_hashtags(hash_keyword)

    # 인기 게시물 정보
    # hashtag_medias_top = cl.hashtag_medias_top(name=hash_keyword, amount=2)

    # 모바일 기준 인기 게시물
    hashtag_medias_top_1 = cl.hashtag_medias_v1(name=hash_keyword, amount=1000, tab_key='top')

    # 모바일 기준 최신 게시물
    # hashtag_medias_2 = cl.hashtag_medias_v1(name=hash_keyword, amount=27, tab_key='recent')

    for i in range(0, len(hashtag_medias_top_1)):
        time.sleep(3)
        text = hashtag_medias_top_1[i].dict()
        pk = text['pk']
        post_id = text['id']
        post_date = text['taken_at'].strftime('%Y-%m-%d %H:%M:%S')
        # 계정명
        user = text['user']['username']
        # 해당 인기 게시물 인스타 계정명
        user_url = "https://www.instagram.com/" + str(user) + "/"
        product = text['code']
        # 해당 게시물 url
        product_url = "https://www.instagram.com/p/" + str(product)
        # 댓글 수
        comment_count = text['comment_count']
        # 좋아요 수
        like_count = text['like_count']
        # 본문 말 ( 해시태그 등 )
        caption_text = text['caption_text']
        file = open('./data/post_data.csv', 'a', encoding='utf-8')
        file.write(
            f'\n"{str(pk)}","{str(post_id)}","{str(hash_keyword)}","{str(user)}","{str(comment_count)}","{str(like_count)}","{str(caption_text)}","{str(post_date)}"')
        file.close()
        # 이미지 추출
        image_list = []
        for j in range(0, len(text['resources'])):
            image_list.append(str(text['resources'][j]['thumbnail_url']))
        # 이미지 저장
        for k, data in enumerate(image_list):
            image_data = open('./data/image_data.csv', 'a', encoding='utf-8')
            random_name = uuid.uuid4()
            print(data)  # 실제 url 임
            image_url = data
            urllib.request.urlretrieve(image_url, "./image_all/"+str(random_name)+".jpg")
            image_data.write(f'\n"{str(pk)}","{str(post_id)}","{str(random_name)}"')
            image_data.close()
        time.sleep(3)
        # 튜플 형태임
        user_data = open('./data/user_data.csv', 'a', encoding='utf-8')
        user_info = cl.user_info_by_username(user)
        user_info_text = user_info.dict()
        # 게시물 수
        user_info_media_count = user_info_text['media_count']
        # 팔로워 수
        user_info_follower_count = user_info_text['follower_count']
        # 팔로잉 수
        user_info_following_count = user_info_text['following_count']
        # 내용
        user_info_biography = user_info_text['biography']
        user_info_biography_2 = user_info_biography.replace(",", "")
        user_info_biography_2 = user_info_biography_2.replace("\n", "")
        user_data.write(f'\n"{str(user)}",{str(user_info_follower_count)},"{str(user_info_following_count)}","{str(user_info_media_count)}"')
        user_data.close()
    send_email()


