mv kaggle.json /root/.kaggle/

# Set permissions
chmod 600 /root/.kaggle/kaggle.json

kaggle competitions download -c generating-graphs-with-specified-properties
unzip -q generating-graphs-with-specified-properties.zip