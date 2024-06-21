import os
import math
import json
import io
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import google.generativeai as genai
import google.ai.generativelanguage as glm
from ultralytics import YOLO
import matplotlib.image as mpimg
import logging




# Initialize FastAPI app
app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Simple in-memory storage for user data, replace with a database in production
users = {}

app = FastAPI()

load_dotenv()

# Configure OpenAI and Google Generative AI
genai.configure(api_key="AIzaSyCgfPJhzV5o1ls5HOOWCZr3ow0REWujfzI")
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history=[])

def get_gemini_response(prompt):
    response = chat.send_message(prompt, stream=True)
    response.resolve()

    # Extract the actual text from the response
    text_parts = [chunk.text for chunk in response]
    full_text = " ".join(text_parts)
    
    return full_text


# Graph generation setup         ## input : Text and image
model_vision = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_vision_response(input_text, image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    if input_text:
        response_graph = model_vision.generate_content({
            'role': "user",
            'parts': [
                {
                    'text': "Giving a picture of a floor plan, draw a network diagram for this image by using real devices like routers, switches, computers and servers.\n"
                            "draw network diagram by using this steps as standard for drawing ,Task: Network Drawing Instruction Please follow these steps to draw the network:\n"
                            "1-Office Connections: office (the number of offices will be determined later by the user) connects to a switch device.\n"
                            "2-Switch Connections: All switches are connected to one switch server.\n"
                            "3-Server Connections: The switch server connects to: One router device. Two server devices.\n"
                            "4-Router device connects to: One firewall.\n"
                            "5-Firewall device connects to: The internet.\n"
                            "6-Return Format: Please ensure the network diagram is represented in JSON format.\n"
                            "The json file should be in the following format: {\"devices\": [{\"id\": \"device-name\", \"type\": \"device-type\", \"location\": \"device-location\", \"connections\": [{\"id\": \"device-connected-to-name\"}]}]}, Make sure to follow this format only."
                },
                {
                    "text": input_text
                },
                {
                    'inline_data': {
                        'data': img_byte_arr,
                        'mime_type': "image/jpeg"
                    },
                }
            ]
        }, generation_config={'temperature': 0})
    else:
        response_graph = model_vision.generate_content(image)

    return response_graph.text

# Define dataset location
dataset_location = r".\Yolo\floorplan-1"
print(dataset_location)

# Load YOLO model
model_yolo = YOLO(r'.\Yolo\best.pt')

# Function to resize image using PIL
def resize_image(image, size):
    return image.resize(size)

# Function to get room information using YOLO
def get_rooms_info(image_source: str) -> list[dict]:
    img = Image.open(image_source)
    img_resized = img.resize((900, 630))  # Resize to (900, 630)
    img_resized.save(image_source)
    img_resized = mpimg.imread(image_source)

    results = model_yolo.predict(image_source, save=True, imgsz=320, conf=0.5)
    rooms_info = []

    for box in results[0].boxes.xywh.cpu():
        x, y, w, h = box
        x_center = x
        y_center = y
        area = w * h

        rooms_info.append({
            "width": w.item(),
            "height": h.item(),
            "location": (x_center.item(), y_center.item()),
            "area": area.item()
        })

    return rooms_info

# Arrange the room information
def arrange_office_info(rooms_info):
    offices = []

    sorted_rooms_info = sorted(rooms_info, key=lambda room: (room["location"][0], room["location"][1]))

    for i, room in enumerate(sorted_rooms_info, start=1):
        office_name = f"Office {i}"
        office_area = room["area"]
        office_location = room["location"]

        office_info = {
            "name": office_name,
            "area": office_area,
            "location": office_location
        }
        offices.append(office_info)

    return offices


# Calculate the centroid location for a list of offices
def find_central_location(offices):
    x_coords = [office['location'][0] for office in offices]
    y_coords = [office['location'][1] for office in offices]
    central_x = sum(x_coords) / len(x_coords)
    central_y = sum(y_coords) / len(y_coords)
    return central_x, central_y

# Determine switch locations by dividing the image into parts
def choose_switch_locations(offices, image_width):
    # Calculate the number of switches needed
    num_offices = len(offices)
    num_switches = math.ceil(num_offices / 5)
    
    # Initialize lists to hold groups of offices and switch locations
    offices_groups = [[] for _ in range(num_switches)]
    switch_locations = {}
    switch_offices = {}

    # Calculate the width of each group section
    group_width = image_width / num_switches

    # Distribute offices into their respective groups
    for office in offices:
        group_index = int(office['location'][0] // group_width)
        offices_groups[group_index].append(office)

    # Calculate the centroid location for each group
    for i, group in enumerate(offices_groups):
        if group:  # Check if the group is not empty
            switch_locations[i] = find_central_location(group)
            switch_offices[i] = group

    return switch_locations, switch_offices

# Visualize the results
def plot_results(image_path, switch_locations, arranged_offices, switch_offices):
    image = mpimg.imread(image_path)
    plt.figure()
    plt.axis('off')
    plt.imshow(image)

    for switch_num, (x1, y1) in switch_locations.items():
        plt.scatter(x1, y1, s=200, c='red', marker='+', edgecolors='black')
        plt.text(x1 + 10, y1 + 10, f"Switch {switch_num + 1}", fontsize=10, color='black')
        
        # Plot lines connecting the switch point to its corresponding offices
        for office in switch_offices[switch_num]:
            office_x, office_y = office["location"]
            plt.plot([x1, office_x], [y1, office_y], color='blue', linestyle='--')

    plt.show()

################################################################
# Endpoint to handle different input scenarios
@app.post("/process_all/")
async def process_all(input_text: str = Form(None), file: UploadFile = File(None)):
    try:
        if input_text and file:
            # Both input text and file are provided (handle graph generation)
            # Save uploaded file
            file_location = f"temp_{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
                
            # Open the image
            image = Image.open(file_location)
            
            # Get Gemini response for graph generation
            graph_response : str = get_gemini_vision_response(input_text, image)
            json_start = graph_response.find('{')
            json_end = graph_response.rfind('}') + 1
            json_str = graph_response[json_start:json_end]
            graph_nodes = json.loads(json_str)
            
            # Save JSON code to a file
            with open("network_diagram.json", "w") as json_file:
                json_file.write(json_str)
            print("JSON code saved to 'network_diagram.json'")   
                    
            # Create graph using networkx
            G = nx.Graph()
            devices = graph_nodes["devices"]
            nodes = []
            edges = []
            for device in devices:
                nodes.append(device["id"])
                for connection in device["connections"]:
                    edges.append((device["id"], connection["id"]))

            G.add_nodes_from(nodes)
            G.add_edges_from(edges)
            
            # Load images for nodes
            img_switch = mpimg.imread(r'.\Grpah\static\switch_image.jpg')
            img_printer = mpimg.imread(r'.\Grpah\static\printer_image.jpg')
            img_Router = mpimg.imread(r'.\Grpah\static\router_image.jpg')
            img_internet = mpimg.imread(r'.\Grpah\static\internet_image.jpg')
            img_Firewall = mpimg.imread(r'.\Grpah\static\firewall_image.jpg')
            img_Server = mpimg.imread(r'.\Grpah\static\server_image.jpg')
            img_pc = mpimg.imread(r'.\Grpah\static\pc_image.jpg')

            # Draw the graph
            pos = nx.spring_layout(G, seed=42)
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            node_images = {}
            for device in devices:
                device_type = device["type"].lower()
                if device_type == "router":
                    node_images[device["id"]] = img_Router
                elif device_type == "switch":
                    node_images[device["id"]] = img_switch
                elif device_type == "server":
                    node_images[device["id"]] = img_Server
                elif device_type in ["pc", "computer"]:
                    node_images[device["id"]] = img_pc
                elif device_type in ["firewall", "firewall device"]:
                    node_images[device["id"]] = img_Firewall
                elif device_type in ["internet", "cloud"]:
                    node_images[device["id"]] = img_internet
                elif device_type == "printer":
                    node_images[device["id"]] = img_printer

            for node, image in node_images.items():
                ax.imshow(image, aspect='auto', extent=(pos[node][0] - 0.1, pos[node][0] + 0.1,
                                                        pos[node][1] - 0.1, pos[node][1] + 0.1))
                ax.text(pos[node][0], pos[node][1] - 0.2, node, ha='center', fontsize=10, color='black')

            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black')
            for subgraph in nx.connected_components(G):
                if len(subgraph) > 1:
                    min_x = min(pos[node][0] for node in subgraph)
                    max_x = max(pos[node][0] for node in subgraph)
                    min_y = min(pos[node][1] for node in subgraph)
                    max_y = max(pos[node][1] for node in subgraph)
                    ax.add_patch(plt.Rectangle((min_x - 0.1, min_y - 0.1), max_x - min_x + 0.2, max_y - min_y + 0.2, fill=False, edgecolor='black'))

            ax.set_xticks([])
            ax.set_yticks([])
         # Show the graph
            plt.savefig('network_graph.png')
            plt.tight_layout()
            plt.close(fig)
            Image.open('network_graph.png')

            plt.savefig(r'.\network_graph.jpg')
            plt.close(fig)

        elif input_text:
           # Handle the case where only text is provided
            text_response = get_gemini_response(input_text)
            print(text_response)
            return JSONResponse(content={"text_response": text_response})

        elif file:
            # Only file is provided (handle image processing - YOLO part)
            # Get room information (dummy data for illustration)
            file_location = f"temp_{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(file.file.read())
                
            # Open the image
            image = Image.open(file_location)
            print(file_location)
            rooms_info = get_rooms_info(file_location)
            # Arrange office information
            arranged_offices = arrange_office_info(rooms_info)
            # Determine switch locations
            image_width = 900
            switch_locations, switch_offices = choose_switch_locations(arranged_offices, image_width)
            # Call the function to plot results
            plot_results(file_location, switch_locations, arranged_offices, switch_offices)
            
            # Print out office and switch information (optional)
            for office in arranged_offices:
                logging.info(f"Name: {office['name']}, Area: {office['area']}, Location: {office['location']}")

            for switch_num, (x1, y1) in switch_locations.items():
                logging.info(f"Switch {switch_num + 1} location: x1={x1}, y1={y1}")
                logging.info(f"Offices for Switch {switch_num + 1}:")
                for office in switch_offices[switch_num]:
                    logging.info(f"Name: {office['name']}, Area: {office['area']}, Location: {office['location']}")

            return JSONResponse(content={"message": "Processed image input successfully", "file_name": file.filename})

        else:
            return JSONResponse(content={"error": "Invalid input. Please provide text, image, or both."}, status_code=400)

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"error": f"Internal Server Error: {str(e)}"}, status_code=500)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Network-Bot API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
