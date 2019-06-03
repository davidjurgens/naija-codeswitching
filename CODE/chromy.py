from selenium import webdriver
import time
import json

'''
This is some test code to grab comments from an article
Here the article link is hard-coded, but the links will instead be read from the
directory containing already cached articles. The local file path would be passed instead
of the URL link
'''
link = "https://www.vanguardngr.com/2018/12/lawmakers-chant-of-lies-rowdiness-total-disgrace-to-nigerians/"
driver = webdriver.Chrome()
driver.get(link)

# Find the div element that holds the DISQUS div
heading = driver.find_element_by_id("disqus_thread")

# Find grab the first iframe which contains the DISQUS threads
iFrame = heading.find_elements_by_tag_name("iframe")

#Open the iframe for the comments with the driver
driver.get(iFrame[0].get_attribute("src"))

#Find the tag containing the comments data
forum = driver.find_element_by_css_selector("script#disqus-forumData")

#Click through until the "Load more comments" button is no longer available
# This is a janky implementation. A better one would use selenium's explicity wait features
# The expression would look similar, except the time module would not used

while True:
    try:
       time.sleep(2)
       driver.find_element_by_css_selector("a.load-more__button").click()
    except:
       print("Done")
       break

# Some comments have not displayed because it has not be verified yet
# This set of expressions finds those comments and makes them visible
moderated = driver.find_elements_by_link_text('Show comment.')
print(list(map(lambda x: x.get_attribute('outerHTML'), moderated)))
for comment in moderated:
  comment.click()

#Since the data is in script tag, this grabs the innerHTML
# which in this case is the json data represent the full set of comments and metadata
# The code converts the json encoded string into a python object 
jObject = json.loads(forum.get_attribute('innerHTML'))

# View output on console
print(list(jObject.keys()))
print(driver.page_source)

'''
The final implementation would probably use Python's multiprocessing module.
The result for each task would be a dictionary representing an articule (key) and the comments (value).
The results from all task would be joined to a master dictionary and saved as a json object

In terms of the list of final. The data is saved in the "vangaurd-raw-data.tar.gz" file.
This would be exctracted and read in via a generator to the implementation of the Pool. 
'''
