from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

class Device:
    def __init__(self, device_type, max_connections, name):
        self.device_type = device_type
        self.max_connections = max_connections
        self.connections = []
        self.name = name

def calculate_score(configuration):
    total_connections = sum(len(device.connections) for device in configuration)
    return total_connections

def generate_alphabet_name(index):
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if index < 26:
        return alphabet[index]
    else:
        return alphabet[index // 26 - 1] + alphabet[index % 26]

def generate_optimal_topology(host_coordinates, num_servers, num_routers):
    devices = []

    hosts = [Device('Host', 1, generate_alphabet_name(i)) for i in range(len(host_coordinates))]
    devices.extend(hosts)

    coordinates = np.array(host_coordinates)
    kmeans = KMeans(n_clusters=num_servers + num_routers, random_state=0).fit(coordinates)
    cluster_assignments = kmeans.labels_

    servers = [Device('Server', 2, f'Server{i+1}') for i in range(num_servers)]
    routers = [Device('Router', 4, f'Router{i+1}') for i in range(num_routers)]
    devices.extend(servers)
    devices.extend(routers)

    for host, cluster in zip(hosts, cluster_assignments):
        cluster_device = devices[len(hosts) + cluster]
        host.connections.append(cluster_device)
        cluster_device.connections.append(host)

    for router in routers:
        available_servers = [server for server in servers if server not in router.connections]

        if available_servers:
            server_to_connect = available_servers[0]
            router.connections.append(server_to_connect)
            server_to_connect.connections.append(router)

    return devices, kmeans

def print_network_configuration(configuration):
    for device in configuration:
        connections_names = ', '.join(neighbor.name for neighbor in device.connections)
        print(f"{device.device_type} {device.name} ({len(device.connections)} connections): {connections_names}")

def plot_network_configuration(configuration, host_coordinates, kmeans):
    cluster_assignments = [device.connections[0].device_type for device in configuration if device.device_type == 'Host']

    for device in configuration:
        for neighbor in device.connections:
            device_index = configuration.index(device)
            neighbor_index = configuration.index(neighbor)

            if device_index < len(host_coordinates) and neighbor_index < len(host_coordinates):
                if device.device_type == 'Host' and neighbor.device_type == 'Server':
                    color = 'green'  # Host to Server connection
                elif device.device_type == 'Host' and neighbor.device_type == 'Router':
                    color = 'orange'  # Host to Router connection
                else:
                    color = 'gray'  # Default connection color

                plt.plot([host_coordinates[device_index][0], host_coordinates[neighbor_index][0]],
                         [host_coordinates[device_index][1], host_coordinates[neighbor_index][1]], color=color)

    for device in configuration:
        device_index = configuration.index(device)
        if device.device_type == 'Router' and device_index < len(host_coordinates):
            plt.scatter(host_coordinates[device_index][0], host_coordinates[device_index][1], label='Router', color='red', marker='s', s=100)
            plt.text(host_coordinates[device_index][0], host_coordinates[device_index][1], device.name, fontsize=8, ha='right', va='bottom')

    for device in configuration:
        device_index = configuration.index(device)
        if device.device_type == 'Server' and device_index < len(host_coordinates):
            cluster_color = 'green' if cluster_assignments[device_index] == 'Server' else 'orange'
            plt.scatter(host_coordinates[device_index][0], host_coordinates[device_index][1], label='Server', color=cluster_color, marker='^', s=100)
            plt.text(host_coordinates[device_index][0], host_coordinates[device_index][1], device.name, fontsize=8, ha='right', va='bottom')

    for x, y, label in zip(*zip(*host_coordinates), [host.name for host in configuration if host.device_type == 'Host']):
        plt.scatter(x, y, label='Host', color='blue', marker='o', s=100)
        plt.text(x, y, label, fontsize=8, ha='right', va='bottom')

    for cluster_center, cluster_color in zip(kmeans.cluster_centers_, ['green', 'orange', 'red']):
        circle = patches.Circle(cluster_center, radius=3, edgecolor=cluster_color, facecolor='none', linestyle='dashed', linewidth=2)
        plt.gca().add_patch(circle)
    
    
    
    server_positions = kmeans.cluster_centers_[:num_servers]
    router_positions = kmeans.cluster_centers_[num_servers:]

    plt.scatter(*zip(*server_positions), label='Server Position', color='purple', marker='x', s=100)
    plt.scatter(*zip(*router_positions), label='Router Position', color='brown', marker='x', s=100)

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Network Configuration')

    plt.show()

if __name__ == "__main__":
    host_coordinates = [(0, 0), (15, 1), (2, 2), (5, 15), (6, 6), (7, 7)]
    num_servers = 1
    num_routers = 2

    optimal_configuration, kmeans = generate_optimal_topology(host_coordinates, num_servers, num_routers)
    print_network_configuration(optimal_configuration)

    server_positions = kmeans.cluster_centers_[:num_servers]
    router_positions = kmeans.cluster_centers_[num_servers:]

    print("\nServer Coordinates:")
    for i, position in enumerate(server_positions):
        print(f"Server {i + 1}: {position}")

    print("\nRouter Coordinates:")
    for i, position in enumerate(router_positions):
        print(f"Router {i + 1}: {position}")

    plot_network_configuration(optimal_configuration, host_coordinates, kmeans)
