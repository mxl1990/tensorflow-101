# -*- coding: utf-8 -*-
import requests
from lxml import etree

proxy = {"https": "127.0.0.1:7757"}


headers = {
	'User-Agent': 'Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19',
}

# url = "https://www.tensorflow.org/api_docs/python/"
url = "https://www.tensorflow.org/api_docs/python/"

print "finish preparing"

z = requests.get(url, headers=headers, proxies=proxy)

print "finish get"

url_sets = []

if z.status_code == 200:
	file_save = open("pachong.txt", "w")
	html = etree.HTML(z.content)
	result = html.xpath('//a[@class="devsite-nav-title gc-analytics-event"]')
	for i in xrange(len(result)):
		pase = result[i].attrib
		if pase["href"].find("python") != -1:
			url_sets.append(pase["href"])
	pase1 = html.xpath('//h1 | //p | //h3')
	for i in xrange(len(pase1)):
		print pase1[i].text
	url = url_sets[0]
	z = requests.get(url, headers=headers, proxies=proxy)
	html = etree.HTML(z.content)
	r = html.xpath('//h1 | /p | //h3')
	print r[0].text
