from bs4 import BeautifulSoup
import requests
import csv

def create_genes(amazonia_list): #'http://amazonia.transcriptome.eu/list.php?section=display&id=388' or 199
    results = requests.get(amazonia_list) 
    list_page = results.text

    soup = BeautifulSoup(list_page, 'html.parser')

    table = soup.find('div', class_='field')
    rows = table.find_all('tr')

    with open('data/genes.csv', 'w') as file:
        fieldnames = []
        for row in rows:
            header_sections = row.find_all('th')
            for item in header_sections:
                fieldnames.append(item.get_text())

        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        writer.writerow(fieldnames)

        for row in rows:
            body_sections = row.find_all('td')
            gene = []
            for i in range(len(body_sections)):
                gene.append(body_sections[i].get_text())
            writer.writerow(gene)

def genes_length():
    with open('data/genes.csv', 'r') as file:
        return len(file.readlines())
        

def id(line_num):
    with open('data/genes.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for line in reader:
            if (reader.line_num == line_num):
                return line["List Item"]

def link(id):
    link = "http://amazonia.transcriptome.eu/expression.php?section=displayData&probeId=" + str(id) + "&series=HBI"
    return link

def abbreviation(id):
    with open('data/genes.csv', 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for line in reader:
            if (line["List Item"] == id): 
                return line["Abbreviation"]


create_genes('http://amazonia.transcriptome.eu/list.php?section=display&id=388')
with open('data/data.csv', 'w') as output:
    fieldnames = ["Gene", "Samples", "Signal", "p-Value"]
    writer = csv.writer(output, delimiter=',', lineterminator='\n')
    writer.writerow(fieldnames)

    for line in range(genes_length()):
        entry = []
        if (id(line) == ""):
            continue
        else:
            results = requests.get(link(id(line)))
            page = results.text
            soup = BeautifulSoup(page, 'html.parser')

            table = soup.find_all('tr')
            for row in table:
                gene = []
                entries = row.find_all('td')
                for i in range(len(entries)):
                    entry = []
                    gene.append(entries[i].get_text())
                    entry.append(abbreviation(id(line)))
                    entry += gene
                writer.writerow(entry)
