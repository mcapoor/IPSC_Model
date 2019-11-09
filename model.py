import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from bs4 import BeautifulSoup
import requests

class Scraper:
    def create_genes(self, amazonia_list): #'http://amazonia.transcriptome.eu/list.php?section=display&id=388' or 199
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
        
        fieldnames = []
        body_sections = []
        gene = []
        rows = []

    def genes_length(self):
        with open('data/genes.csv', 'r') as file:
            return len(file.readlines())
            
    def id(self, line_num):
        with open('data/genes.csv', 'r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for line in reader:
                if (reader.line_num == line_num):
                    return line["List Item"]

    def link(self, id):
        link = "http://amazonia.transcriptome.eu/expression.php?section=displayData&probeId=" + str(id) + "&series=HBI"
        return link

    def abbreviation(self, id):
        with open('data/genes.csv', 'r') as file:
            reader = csv.DictReader(file, delimiter=',')
            for line in reader:
                if (line["List Item"] == id): 
                    return line["Abbreviation"]

    def create_data(self):
        with open('data/data.csv', 'w') as output:
            fieldnames = ["Gene", "Samples", "Signal", "p-Value"]
            writer = csv.writer(output, delimiter=',', lineterminator='\n')
            writer.writerow(fieldnames)

            for line in range(Scraper.genes_length(self)):        
                results = requests.get(Scraper.link(self, Scraper.id(self, line)))
                page = results.text
                soup = BeautifulSoup(page, 'html.parser')

                table = soup.find_all('tr')
                for row in table:
                    gene = []
                    entries = row.find_all('td')
                    for i in range(3):
                        gene.append(entries[i].get_text())
                    entry = []
                    entry.append(Scraper.abbreviation(self, Scraper.id(self, line)))
                    entry += gene

                    writer.writerow(entry)

        fieldnames = []
        table = []
        gene = []
        entry = []


    def main(self):
        Scraper.create_genes(self, 'http://amazonia.transcriptome.eu/list.php?section=display&id=388')
        Scraper.create_data(self)

class Model:
    def nonlin(self, X, deriv = False):
        if (deriv == True):
            return X * (1 - X)
        else:
            return 1 / (1 + np.exp(-X))

    def init_data(self):
        scraper = Scraper()
        scraper.main()

        with open('data/data.csv', 'r') as file:
            reader = csv.reader(file, delimiter=',')
            headers = next(reader)
            data = list(reader)
            data = np.array(data, dtype=object)
            
            abbreviation = data[:,0]
            cell = data[:,1]
            signal = data[:,2]
            p_value = data[:,3]

            onehot_encoder = OneHotEncoder(sparse=False)
        
            abbreviation = abbreviation.reshape(len(abbreviation), 1)
            abbreviation_encoded = onehot_encoder.fit_transform(abbreviation)

            cell = cell.reshape(len(cell), 1)
            cell_encoded = onehot_encoder.fit_transform(cell)

            abbreviation_encoded = np.array(abbreviation_encoded)
            cell_encoded = np.array(cell_encoded)
            p_value = np.array(p_value)
            signal = np.array(signal)

            input = np.column_stack((abbreviation_encoded, cell_encoded, p_value))
            output = signal
            
            data = []
            abbreviation = []
            cell = []
            abbreviation_encoded = []
            cell_encoded = []
            p_value = []

        return input, output

    def error(self, out, Y):
        sum = 0
        error = (1/2) * ((Y - out) ** 2)
        
        rows = np.shape(error)[0]
        columns = np.shape(error)[1]
        for row in range(rows):
            for column in range(columns):
                sum += out[row][column]

        mean = sum / (rows * columns)
        return round(mean * 100, 2)

    def layer_out(self, layer_input, layer_weights):
        net = np.dot(layer_input, layer_weights)
        out = nonlin(net)
        return net, out

    def update_weights(self, layer_input, layer_weights, Y):
        net, out = layer_out(layer_input, layer_weights)
        self.weight_delta = -(Y - out) * (net * (1 - net)) * layer_input
        
        net = []
        out = []

        return weight_delta

    def main(self):
        X, Y = Model.init_data(self)
        for epoch in range(100000):
            print("Training Epoch:", epoch, "-", (epoch/10000) * 100, "%")

            inputs = np.shape(X)[1]
            hidden_neurons = 2 * inputs
            output_neurons = np.shape(Y)[0]

            input_weights = np.multiply(2, np.random.rand(inputs, hidden_neurons)) - 1
            hidden_weights = 2 * (np.random.rand(hidden_neurons, output_neurons)) - 1
            output_weights = 2 * (np.random.rand(output_neurons, 0)) - 1

            l1 = Model.layer_out(self, X, input_weights)
            input_weights += Model.update_weights(self, X, input_weights)
            
            h1 = Model.layer_out(self, l1, hidden_weights)
            first_hidden_weights += Model.update_weights(self, l1, hidden_weights)

            output = Model.layer_out(self, h1, output_weights)
            output_weights += Model.update_weights(self, h1, output_weights)
            
        print("Predicted:\n", output)
        print("\nActual:\n", Y)
        print("\n Mean Error:\n", Model.error(self, output, Y),"%")

model = Model()
model.main()