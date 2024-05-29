# post_data = open('./data/post_data.csv','w')
# post_data.write("hashtag,username,comment_count,like_count,caption_text,media_count,follower_count,following_count,"
#            "image_name,post_date")
# post_data.close()
#
# user_data = open('./data/user_data.csv','w')
# user_data.write("username,follower_count,following_count,media_count")
# user_data.close()

# file = open('./data/instagram_post_data', 'w')
# file.write("hashtags,instagram_url,username,insta_followers,post_url,like_count,comment_count,user_info_biography_2"+'\n')
# file.close()

# hashtag
# username
# comment_count
# like_count
# caption_text
# media_count
# follower_count
# following_count
# image_name
# post_date
# image_data = open('./data/image_data.csv','w')
# image_data.write("pk,post_id,image_name")
# image_data.close()
#
# instagram_post_data = open('./data/post_data.csv','w')
# instagram_post_data.write("pk,post_id,username,hashtag,comment_count,like_count,caption_text,post_date")
# instagram_post_data.close()

predict_data = open('./data/predict_data.csv', 'w')
predict_data.write("username,follower_count,following_count,media_count,hashtag,image_name")
predict_data.close()