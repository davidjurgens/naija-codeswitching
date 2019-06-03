import time
import os
import csv
import glob
import sys, traceback
import json
import logging

def parse_comments(comment_data):
		comment_result = {}
		
		if not comment_data["underlying"]:
			return {}

		if not comment_data["underlying"]["response"]:
			return {}

		posts = comment_data["underlying"]["response"]["underlying"]["posts"]

		if not posts["value"]:
			return {}

		for post in posts["value"]:
			comment_result[post["underlying"]["id"]["value"]] = dict(createdAt=post["underlying"]["createdAt"]["value"], 
				message=post["underlying"]["message"], author=post["underlying"]["author"]["underlying"]["name"]["value"], 
				depth=post["underlying"]["depth"]["value"], points=post["underlying"]["points"]["value"], 
				profile=post["underlying"]["author"]["underlying"]["profileUrl"]["value"])

			if post["underlying"]["depth"]["value"] > 0:
				comment_result[post["underlying"]["id"]["value"]]["parent"] = post["underlying"]["parent"]["value"]
			else:
				comment_result[post["underlying"]["id"]["value"]]["parent"] = post["underlying"]["id"]["value"]
		return comment_result


if __name__ == "__main__":
	""" Run one with argument:
		<1> : directory containing comment data
	"""

	comments_directory = sys.argv[1]

	directories = glob.glob(comments_directory+"*/")
	
	num_articles_with_comments = 0
	num_articles = 0

	for directory in directories:
		print("Processing " + directory)

		comment_data = {}

		for file in glob.glob(directory+"*.json"):
			with open(file, 'r') as f:
				compiled_comments = json.load(f)
				for article in compiled_comments:
					num_articles += 1
					for uri, comments in article.items():
						comment_data[uri] = parse_comments(comments)
						if comment_data[uri]:
							num_articles_with_comments += 1

		with open(os.path.basename(directory)+'.json', 'w') as f:
			f.write(json.dumps(comment_data))


	print("{}/{} articles have comments".format(num_articles_with_comments, num_articles))
