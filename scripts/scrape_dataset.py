import requests
import urllib2
from lxml import html

#page = requests.get("http://gpcr.biocomp.unibo.it/~emidio/I-Mutant2.0/dbMut3D.html", timeout=5)
page = urllib2.urlopen("http://gpcr.biocomp.unibo.it/~emidio/I-Mutant2.0/dbMut3D.html")
print "here"
tree = html.parse(page)
print "here2"
fw = open('./data/imutant_dataset.txt', 'w')

table = tree.xpath("//html/body/table[2]/tbody/tr/td[2]/center/form/table[1]/tbody/tr/td/div/p[2]/table/tbody")
print table
for row in table.xpath('/tr'):
  for cell in row.xpath('/td'):
    value = cell.xpath('/div/text()')
    print value
    fw.write(cell.xpath('/div/text()') + ' ')
  fw.write('\n')
