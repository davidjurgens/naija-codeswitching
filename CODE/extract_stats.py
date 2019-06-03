import pickle
import os
import glob
import sys

import json

if __name__ == "__main__":
	article_directory = sys.argv[1]
	comment_directory = sys.argv[2]
	websites = glob.glob(article_directory+'/*.tsv')

	comment_files = glob.glob(comment_directory+'/*.json')
	num_articles = 0
	num_comments = 0

	for website in websites:
		with open(website, 'r') as f:
			tsv_reader = csv.reader(f, delimiter="\t")
			num_articles += sum(1 for row in tsv_reader)

	for comment_files in comment_filess:
		with open(comment_files, 'r') as f:
			comments = json.load(f)

			for article,comment_data in comments.items():
				if comment_data:
					num_comments += sum(1 for post in comment_data.items())


	print("{} total articles".format(num_articles))
	print("{} total comments".format(num_comments))


			