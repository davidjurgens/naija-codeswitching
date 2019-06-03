from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import multiprocessing
from multiprocessing import Pool, TimeoutError
from selenium.webdriver.chrome.options import Options
import time
import os
import csv
import glob
import sys, traceback
import json
import logging


class Logger(object):
	logger = None
	def myLogger(self):
		if None == self.logger:
			self.logger=multiprocessing.log_to_stderr()
			self.logger.setLevel(logging.DEBUG)
			#handler = logging.StreamHandler()
			#handler.setFormatter(logging.Formatter("%(levelname)s: %(asctime)s - %(process)s - %(message)s"))

		return self.logger

# TODO: Include loggers and remove test variables
# TODO: Need thread-safe loggers. Current implementation is not pickleable
# Also should work for Ubuntu
class CommentDriver(webdriver.Chrome):
	"""	
	class CommentDriver

	This class extends the Chrome driver and is use to execute a driver that grabs
	disqus comments

	----------------
	Attributes:
	----------------
	urls (type: List[String])= the list of urls from which to grab the comments
	path (type: String) = the path to the directory where the cached articles are saced
	args (type: List[String]) = options to be used to initialize the Chrome driver
	disqus_div_id (type = String) = the #id for the 'div' tag that contains the Disqus iframe
	data_script_id (type = String) = the #id for the 'script' tag that contains the thread data

	A class instance is the driver. So en lieu of calling driver.get(), call self.get()
	
	----------------
	Methods:
	----------------
	See Docstrings
	"""

	def __init__(self, path, args = None):
		self.url = None
		self.disqus_div_id = None
		self.data_script_id = None

		#Overloading the parent constructor
		if (args):
			chrome_options = Options()
			for arg in args:
				chrome_options.add_argument("--{}".format(arg))
			super(CommentDriver, self).__init__(path, chrome_options=chrome_options)
		else:
			super(CommentDriver, self).__init__(path)

	'''
	params: json object containing disqus thread data
	Parses json and extracts essential information 
	'''

	def parse_comments(self, comment_data):
		comment_result = {}
		posts = comment_data["response"]["posts"]

		for post in posts:
			comment_result[post["id"]] = dict(createdAt=post["createdAt"], message=post["message"], author=post["author"]["name"], 
				depth=post["depth"], points=post["points"])
			if post["depth"] > 0:
				comment_result[post["id"]]["parent"] = post["parent"]
			else:
				comment_result[post["id"]]["parent"] = post["id"]
		return comment_result

	def config(self, url, disqus_div_id = None, data_script_id = None):
		'''
		params: url, disqus_div_id, data_script_id
		Set the configuration for the driver. Also formats the string read from file.
		'''
		self.url =  "file://"+url
		self.url = self.url.replace("%25", "%")
		self.url = self.url.replace("%", "%25")
		self.disqus_div_id = disqus_div_id
		self.data_script_id = data_script_id
		print(url)


	def run(self):
		'''
		params: None

		Run the driver for a link. Loads the page, grabs the iframe, and scrapes
		the comment data. 

		TODO: Timeout could be handle better than just hard-coding a time-interval

		returns a tuple containing the url and the comment data

		'''
		try:
			self.get(self.url)
			# Find the div element that holds the DISQUS div
			disqus_div = self.find_element_by_id(self.disqus_div_id)
			# Find grab the first iframe which contains the DISQUS threads
			disqus_iframe = disqus_div.find_elements_by_tag_name("iframe")
			#Open the iframe for the comments with the driver
			self.get(disqus_iframe[0].get_attribute("src"))
			#Find the tag containing the comments data
			forum = self.find_element_by_css_selector("script#{}".format(self.data_script_id))
			print("Checking for loading buttons!")
			while True:
				try:
					time.sleep(2)
					self.find_element_by_css_selector("a.load-more__button").click()
				except:
					print("Done")
					break

			try:
				moderated = self.find_elements_by_link_text('Show comment.')
				#print(list(map(lambda x: x.get_attribute('outerHTML'), moderated)))
				for comment in moderated: 
					comment.click()
			except:
				print("Nothing to click!")

			jObject = json.loads(forum.get_attribute('innerHTML'))
			self.close()
			return (self.url, self.parse_comments(jObject))
		except:
			print("This is the problem child: {}\nERROR:{}".format(self.url, sys.exc_info()[0]))
			print("Exception in user code:")
			print('-'*60)
			traceback.print_exc(file=sys.stdout)
			print('-'*60)
			return None


class PoolManager(object):
	''''
	class PoolManager

	This class defines the structure that will manage the ForkJoin pool created by the
	multiprocessing library.

	----------------
	Attributes:
	----------------
	urls (type: List[String])= the list of urls from which to grab the comments
	path (type: String) = the path to the directory where the cached articles are saced
	args (type: List[String]) = options to be used to initialize the Chrome driver
	disqus_div_id (type = String) = the #id for the 'div' tag that contains the Disqus iframe
	data_script_id (type = String) = the #id for the 'script' tag that contains the thread data

	----------------
	Methods:
	----------------
	See Docstrings
	'''

	def __init__(self, logger,path=None, article_directory_path=None):
		#self.logger = logger.myLogger()
		''''self.urls = ["file:///Volumes/G-DRIVE/DataMining_Project/Data/rawdata/vangaurd/data/vanguardngr_editorial_electronic-voting-possible-if.html",
		"file:///Volumes/G-DRIVE/DataMining_Project/Data/rawdata/vangaurd/data/vanguardngr_editorial_elevating-electricity-statistics.html",
		"file:///Volumes/G-DRIVE/DataMining_Project/Data/rawdata/vangaurd/data/vanguardngr_editorial_fiscal-federalism-viability-states.html",
		"file:///Volumes/G-DRIVE/DataMining_Project/Data/rawdata/vangaurd/data/vanguardngr_editorial_from-mdgs-to-sdgs-meeting-the-new-target.html",
		"file:///Volumes/G-DRIVE/DataMining_Project/Data/rawdata/vangaurd/data/vanguardngr_editorial_gov-el-rufais-feud-with-beggars.html",
		"file:///Volumes/G-DRIVE/DataMining_Project/Data/rawdata/vangaurd/data/vanguardngr_editorial_herdsmen-attacks-trigger-famine.html"]
		'''
		self.result = {}
		self.urls = []

		with open(path) as f:
			tsv_reader = csv.reader(f, delimiter="\t")
			next(tsv_reader)
			for article in tsv_reader:
				self.result[article[6]] = dict(datetime=article[0], section=article[1], title=article[2], 
					author=article[3], text=article[4].replace('By ', '').replace(article[3], ''), source=article[5])
				self.urls.append(article_directory_path + article[6])

		#self.logger.setLevel(logging.INFO)

	def driver_config(self, path=None, args=None, disqus_div_id=None, data_script_id=None):
		"""
		params: path, args, disqus_div_id, data_script_id
		Sets the parameters for the chrome driver configuratiosn
		"""
		print("Setting driver configuration properties")
		self.path = path #"/usr/local/bin/chromedriver"
		self.args = ["disable-default-apps","disable-gpu","disable-extensions","no-default-browser-check", "headless", "no-sandbox", "disable-dev-shm-usage"]
		self.disqus_div_id = "disqus_thread" #disqus_div_id
		self.data_script_id =  "disqus-threadData" #data_script_id
		print("\n*****\tDRIVER CONFIGURATIONS\t****")
		print("PATH_TO_DRIVER \t\t\t===> {}".format(self.path))
		print("--headless\t\t\t==> true")
		print("--no-sandbox\t\t\t==> true")
		print("--disable-dev-shm-usage\t\t==> true\n")
		print("*****\tDISQUS PROPERTIES\t*****")
		print("DISQUS DIV ID: {}".format(self.disqus_div_id))
		print("DISQUS FORUM DATA TAG ID: {}".format(self.data_script_id))

	def get(self, url):
		"""
		params: url
		Runs the driver to retrieve the comments for a given url

		returns a tuple containing the url and the comment data  
		"""
		print("PID {} ~ SCANNING: {}".format(os.getpid(), url))
		comment_driver = CommentDriver(self.path, self.args)
		comment_driver.config(url, self.disqus_div_id, self.data_script_id)
		result = comment_driver.run()
		print("PID {} ~ COMMENT RETRIEVED: {}".format(os.getpid(), url))
		del(comment_driver)
		return result


	def save_to_disk(self, filename, list_of_tups):
		"""
		params: filename (type: String), list_of_tups (type: List[Tuples])
		Saves the result from entire jobs to disk

		"""
		print("PID {} ~ SAVING TO DISK".format(os.getpid()))

		for comment_data in list_of_tups:
			try:
				self.result[os.path.basename(comment_data[0])]["comments"] = comment_data[1]
			except:
				print("No data for file")

		with open(filename, "w") as f:
			encodingString = json.dumps(self.result, indent=4)
			f.write(encodingString)

	def run(self):
		"""
		params: None

		Initializes the Pool() and passess the urls for processing

		returns a list of tuples,where each tuple represent an article uri and its disqus comments
		"""
		with Pool() as pool:
			print("\n\n***STARTING POOLED SCRAPER***")
			print("(1) Number of processes: {}".format(pool._processes))
			print("(2) CPU Count: {}\n".format(multiprocessing.cpu_count()))
			comments = pool.map(self.get, self.urls[:10])
		return comments



if __name__ == "__main__":

	"""
	Run this file with three arguments:

	python pooled.py <file_directory_path> <file_path_to_save_results> <path_to_chrome_driver>

	(1) <article_data path>:
		- Pass the file path in quotes
		Exmp: python pooled.py "/Volumes/G-DRIVE/DataMining_Project/Data/vanguard"

	(2)<articles directory>
		- Pass the directory containing the articles in quotes

	(3) <file_path_to_save_results>:
		- absolute path to save the results. Also in quotes:
		Exmp: python pooled.py "/Volumes/G-DRIVE/DataMining_Project/Data/vanguard" "/Volumes/G-DRIVE/DataMining_Project/Data/vanguard_comments"

	(4) <path_to_chrome_driver>:
		- Absolute path to your chrome_driver install. Also in quotes:
		Exmp: python pooled.py "/Volumes/G-DRIVE/DataMining_Project/Data/vanguard" "/Volumes/G-DRIVE/DataMining_Project/Data/vanguard_comments" "/Volumes/G-DRIVE/chromedriver"
		
	"""
	logger = Logger()

	#Passed in arguments for the directory path and the path to save the results
	file_to_save = sys.argv[3]
	articles_directory = sys.argv[2]
	articles_file = sys.argv[1]
	chrome_path = sys.argv[4]

	# Initialize the PoolManager()
	pm = PoolManager(logger, articles_file, articles_directory)

	# Set the configuration parameters
	pm.driver_config(path=chrome_path)

	#Run the collection
	results = pm.run()

	#Save to disk
	pm.save_to_disk(file_to_save, results)

