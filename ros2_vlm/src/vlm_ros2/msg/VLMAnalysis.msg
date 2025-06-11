# VLM Analysis Result Message
# Contains the analysis output from the Vision Language Model

# Header with timestamp
std_msgs/Header header

# The text analysis/description from VLM
string text

# The prompt/question that was asked
string prompt

# Confidence score (0.0 to 1.0)
float32 confidence

# Processing time in seconds
float32 processing_time

# Model name used for analysis
string model_name

# Optional: Region of interest if analyzing specific area
sensor_msgs/RegionOfInterest roi