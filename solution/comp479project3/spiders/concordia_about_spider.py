import scrapy
from bs4 import BeautifulSoup
from scrapy.exceptions import CloseSpider
from urllib.parse import urlparse
import re
import os
from pathlib import Path


class ConcordiaAboutSpider(scrapy.Spider):

    name = 'about_crawl'
    allowed_domains = ['concordia.ca']
    start_urls = ['https://www.concordia.ca/about.html']
    max_document_size = 200  # Number of links the crawler will extract
    document_counter = 0

    def parse(self, response):
        if self.document_counter <= self.max_document_size:
            self.document_counter += 1
            # Passing the response body to BeautifulSoup
            soup = BeautifulSoup(response.body.decode('utf-8'), 'html.parser')
            self.extract_content(response.url, soup)
            # Extracting links from main content, ignored navbar
            links = soup.find(id='content-main').find_all('a', href=re.compile(r'.*html$'))  # TODO improve regex
            # links = response.css('a::attr(href)')
            # print(links)
            for link in links:
                # Recursive call to extra data in breadth first search (BFS) order -> see settings.py line 23
                yield response.follow(link.get('href'), callback=self.parse)
        else:
            raise CloseSpider('max_document_exceeded')

    @staticmethod
    def extract_content(url, soup):
        sub_titles = urlparse(url).path.split('.')[0].split('/')
        title = '_'.join(sub_titles)[1:]
        # print(title)
        tags = ['p', 'span', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'th', 'td']
        content = ''
        for tag in tags:
            content += '\n' + '\n'.join([txt.text for txt in soup.find(id='content-main').find_all(tag)])
        # print(content)
        ConcordiaAboutSpider.write_content_to_file(url, title, content)

    @staticmethod
    def write_content_to_file(url, title, content):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(Path(current_dir).parent.parent, 'extracted_files')
        try:
            os.makedirs(output_dir)
        except OSError:
            pass
        with open(output_dir + '/' + title + '.txt', 'w') as file:
            data = {'title': title, 'url': url, 'content': content}
            file.write(str(data))
