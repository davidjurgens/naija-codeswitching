import os
import glob
import json
import sys
import csv
from bs4 import BeautifulSoup

'''
run with 2 arguments:
(1) path to article data
(2) path to save JSON result
'''

csv.field_size_limit(sys.maxsize)

def parse_article(url):
	soup = BeautifulSoup(open(url), "html.parser")
	p_tags = soup.find_all('p')
	result = ''
	for tag in p_tags:
		result += tag.get_text()

	return result.replace('Share ', '').replace('withFacebookMessengerMessengerWhatsAppTwitterEmailCopy', '').replace('dis', '').replace('link', '')

if __name__ == "__main__":
	article_directory = sys.argv[1]
	path_to_save = sys.argv[2]

	result = ''

	path = os.path.join(article_directory, '*.html')

	urls = glob.glob(path)

	i = 0
	for url in urls:
		result += parse_article(url)
		i += 1
		print("{} articles processed".format(i))

	with open(path_to_save, "w") as f:
		f.write(result.encode('utf-8'))
