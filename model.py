import time
import csv

import requests
import numpy as np

from bs4 import BeautifulSoup
from sklearn.preprocessing import OneHotEncoder

class Scraper:
    def create_genes(self, list_id):
        list_page = requests.get('http://amazonia.transcriptome.eu/list.php?section=display&id=' + list_id).text

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

        fieldnames = []
        body_sections = []
        gene = []
        rows = []

    def genes_length(self):
        with open('data/genes.csv', 'r') as file:
            return len(file.readlines())

    def label_id(self, line_num, list_id):
        with open('data/genes.csv', 'r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for line in reader:
                if reader.line_num == line_num:
                    return line["List Item"]

    def link(self, _id):
        link = "http://amazonia.transcriptome.eu/expression.php?section=displayData&probeId=" + str(_id) + "&series=HBI"
        return link

    def abbreviation(self, _id):
        with open('data/genes.csv', 'r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for line in reader:
                if line["List Item"] == _id:
                    return line["Abbreviation"]

    def create_data(self, list_id):
        with open('data/data.csv', 'w') as output:
            fieldnames = ["Gene", "Samples", "Signal", "p-Value"]
            writer = csv.writer(output, delimiter=',', lineterminator='\n')
            writer.writerow(fieldnames)

            for line in range(Scraper.genes_length(self, list_id)):
                results = requests.get(Scraper.link(self, Scraper.label_id(self, line, list_id))).text
                soup = BeautifulSoup(results, 'html.parser')

                table = soup.find_all('tr')
                for row in table:
                    gene = [Scraper.abbreviation(self, Scraper.label_id(self, line, list_id), list_id)]
                    entries = row.find_all('td')
                    for i in range(3):
                        gene.append(entries[i].get_text())
                    writer.writerow(gene)

        fieldnames = []
        table = []
        gene = []

    def main(self, list_id):
        Scraper.create_genes(self, list_id)
        Scraper.create_data(self, list_id)

class Model:
    def nonlin(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def preload_weights(self):
        response = input("Preload weights?")
        if 'yes' or 'y' in response.lower():
            with open("data/weights.csv", "r") as file:
                reader = csv.reader(file).read()
                data = list(next(next(reader)))
                data = np.array(data, dtype=float)

                input_weights = data[0]
                hidden_weights = data[1]
                output_weights = data[2]

                return input_weights, hidden_weights, output_weights

    def init_weights(self, x, y):
        response = input("Preload weights?")
        if 'yes' or 'y' in response.lower():
            input_weights, hidden_weights, output_weights = Model.preload_weights(self)
        else:
            inputs = np.shape(x)[1]
            hidden_neurons = 2 * inputs
            output_neurons = np.shape(y)[0]

            input_weights = 2 * np.random.rand(inputs, hidden_neurons) - 1
            hidden_weights = 2 * np.random.rand(hidden_neurons, output_neurons) - 1
            output_weights = 2 * np.random.rand(output_neurons, 0) - 1

        return input_weights, hidden_weights, output_weights

    def init_data(self):
        if 'yes' or 'y' in input("Generate new database files?").lower():
            print("Creating Data file")
            scraper = Scraper()
            amazonia_list_id = '388'
            scraper.main(amazonia_list_id)

        print("Initialising Data...\n")
        time.perf_counter()
        with open('data/data.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            headers = next(reader)
            data = list(reader)
            data = np.array(data, dtype=object)

            abbreviation = data[:, 0]
            cell = data[:, 1]
            signal = data[:, 2]
            p_value = data[:, 3]

            onehot_encoder = OneHotEncoder(sparse=False)

            abbreviation = abbreviation.reshape(len(abbreviation), 1)
            abbreviation = onehot_encoder.fit_transform(abbreviation)

            cell = cell.reshape(len(cell), 1)
            cell = onehot_encoder.fit_transform(cell)

            abbreviation = np.array(abbreviation)
            cell = np.array(cell)
            p_value = np.array(p_value)
            signal = np.array(signal)

            value_input = np.column_stack((abbreviation, cell, p_value))
            value_output = signal

            data = []
            abbreviation = []
            cell = []
            p_value = []

        print("Data Initialised in {time.perf_counter()} seconds\n")
        return value_input, value_output

    def error(self, out, y):
        total = 0
        error = (1/2) * ((y - out) ** 2)

        rows = np.shape(error)[0]
        columns = np.shape(error)[1]
        for row in range(rows):
            for column in range(columns):
                total += out[row][column]

        mean = total / (rows * columns)
        return round(mean * 100, 2)

    def layer_out(self, layer_input, layer_weights):
        net = np.dot(layer_input, layer_weights)
        out = Model.nonlin(self, net)
        return net, out

    def update_weights(self, layer_input, layer_weights, y):
        net, out = Model.layer_out(self, layer_input, layer_weights)
        weight_delta = -(y - out) * (net * (1 - net)) * layer_input

        net = []
        out = []

        return weight_delta

    def write_weights(self, i_w, h_w, o_w):
        with open('data/weights.csv', 'w') as file:
            writer = csv.writer(file, lineterminator='\n')
            writer.writerow(["Input Weights", "Hidden Weights", "Output Weights"])
            writer.writerow(i_w, h_w, o_w)

    def main(self):
        x, y = Model.init_data(self)
        input_weights, hidden_weights, output_weights = Model.init_weights(self, x, y)

        for epoch in range(100000):
            print("Training Epoch: {epoch}, - {(epoch / 1000) * 100}%")
            time.perf_counter()

            l_1 = Model.layer_out(self, x, input_weights)
            input_weights += Model.update_weights(self, x, input_weights, y)

            h_1 = Model.layer_out(self, l_1[1], hidden_weights)
            first_hidden_weights += Model.update_weights(self, l_1, hidden_weights, y)

            output = Model.layer_out(self, h_1[1], output_weights)
            output_weights += Model.update_weights(self, h_1, output_weights, y)

            print("Epoch {epoch}/1000 completed in {time.perf_counter()} seconds\n")

        print("Mean Error:", Model.error(self, output, y), "%")
        Model.write_weights(self, input_weights, hidden_weights, output_weights)

model = Model()
model.main()
