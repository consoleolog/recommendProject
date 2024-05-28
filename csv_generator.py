post_data = open('./data/post_data.csv','w')
post_data.write("writer,hashtag,likes,post_date,photos")
post_data.close()

user_data = open('./data/user_data.csv','w')
user_data.write("writer,following,followers,post_count")
user_data.close()