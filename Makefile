format:
	ruff check --fix .
	ruff format .
	pyright

datasets:
	# Create a directory for the dataset
	# Check if the directory already exists
	if [ ! -d "Datasets" ]; then \
		echo "Creating Datasets directory"; \
		mkdir Datasets; \
	else \
		echo "Datasets directory already exists"; \
	fi
	cd Datasets
	# Download the dataset: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
	# Check if the download was successful
	if [ ! -f cifar-10-python.tar.gz ]; then \
		echo "Download failed"; \
		exit 1; \
	fi
	# Move the downloaded file to the Datasets directory
	mv cifar-10-python.tar.gz Datasets/
	# Extract the dataset
	tar xvfz Datasets/cifar-10-python.tar.gz
	# Remove the tar file
	rm Datasets/cifar-10-python.tar.gz
	# Move the extracted files to the Datasets directory
	mv cifar-10-batches-py Datasets/

a1:
	python3 Assignment1/Assignment1.py