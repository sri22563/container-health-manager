import time
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Service Metrics Collection (Simulation) ---

class ServiceMetrics:
    def __init__(self, cpu_usage, memory_usage, request_latency, error_rate):
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.request_latency = request_latency
        self.error_rate = error_rate

    def __repr__(self):
        return (f"Metrics(CPU: {self.cpu_usage:.1f}%, Mem: {self.memory_usage:.1f}%, "
                f"Latency: {self.request_latency:.1f}ms, Errors: {self.error_rate:.1f}%)")

def simulate_metrics(base_metrics, variance=5, error_spike_prob=0.1, latency_spike_prob=0.1):
    cpu = max(0, min(100, base_metrics.cpu_usage + random.uniform(-variance, variance)))
    memory = max(0, min(100, base_metrics.memory_usage + random.uniform(-variance, variance)))
    latency = max(1, base_metrics.request_latency + random.uniform(-variance * 2, variance * 2))
    errors = max(0, min(100, base_metrics.error_rate + random.uniform(-variance / 2, variance / 2)))

    if random.random() < error_spike_prob:
        errors += random.uniform(10, 30)
    if random.random() < latency_spike_prob:
        latency += random.uniform(50, 200)

    return ServiceMetrics(cpu, memory, latency, errors)

# --- 2. Health Assessment ---

class ServiceHealth:
    def __init__(self, is_healthy, status_message="OK"):
        self.is_healthy = is_healthy
        self.status_message = status_message

    def __repr__(self):
        return f"Health(Healthy: {self.is_healthy}, Status: '{self.status_message}')"

def assess_health(metrics, thresholds):
    is_healthy = True
    messages = []

    if metrics.cpu_usage > thresholds['cpu_high']:
        is_healthy = False
        messages.append(f"High CPU usage ({metrics.cpu_usage:.1f}%)")
    if metrics.memory_usage > thresholds['memory_high']:
        is_healthy = False
        messages.append(f"High Memory usage ({metrics.memory_usage:.1f}%)")
    if metrics.request_latency > thresholds['latency_high']:
        is_healthy = False
        messages.append(f"High Latency ({metrics.request_latency:.1f}ms)")
    if metrics.error_rate > thresholds['error_high']:
        is_healthy = False
        messages.append(f"High Error rate ({metrics.error_rate:.1f}%)")

    status_message = "OK" if is_healthy else ", ".join(messages)
    return ServiceHealth(is_healthy, status_message)

# --- 3. Representing Services ---

class Microservice:
    def __init__(self, name, base_metrics, health_thresholds, dependencies=None):
        self.name = name
        self.base_metrics = base_metrics
        self.current_metrics = simulate_metrics(base_metrics)
        self.health_thresholds = health_thresholds
        self.current_health = assess_health(self.current_metrics, self.health_thresholds)
        self.dependencies = dependencies if dependencies is not None else []
        self.state = "running"

    def update_metrics(self):
        self.current_metrics = simulate_metrics(self.base_metrics)

    def update_health(self):
        self.current_health = assess_health(self.current_metrics, self.health_thresholds)
        if not self.current_health.is_healthy and self.state == "running":
            print(f"Service '{self.name}' detected as unhealthy.")
            self.state = "unhealthy"
        elif self.current_health.is_healthy and self.state != "running":
            print(f"Service '{self.name}' is now healthy.")
            self.state = "running"

    def __repr__(self):
        return f"Service(Name: {self.name}, State: {self.state}, Health: {self.current_health.status_message})"

# --- 4. Setup Services ---

frontend_metrics = ServiceMetrics(cpu_usage=20, memory_usage=30, request_latency=50, error_rate=1)
user_service_metrics = ServiceMetrics(cpu_usage=15, memory_usage=25, request_latency=30, error_rate=0.5)
product_service_metrics = ServiceMetrics(cpu_usage=18, memory_usage=28, request_latency=40, error_rate=0.8)
order_service_metrics = ServiceMetrics(cpu_usage=25, memory_usage=35, request_latency=60, error_rate=1.5)

common_thresholds = {
    'cpu_high': 70,
    'memory_high': 80,
    'latency_high': 200,
    'error_high': 5
}

frontend = Microservice("Frontend", frontend_metrics, common_thresholds, dependencies=["User Service", "Product Service"])
user_service = Microservice("User Service", user_service_metrics, common_thresholds)
product_service = Microservice("Product Service", product_service_metrics, common_thresholds)
order_service = Microservice("Order Service", order_service_metrics, common_thresholds, dependencies=["User Service", "Product Service"])

services = {
    frontend.name: frontend,
    user_service.name: user_service,
    product_service.name: product_service,
    order_service.name: order_service
}

# --- 5. Generate Training Data ---

def generate_historical_data(num_samples_per_service=200):
    data = []
    for service_name, service in services.items():
        for _ in range(num_samples_per_service):
            metrics = simulate_metrics(service.base_metrics, variance=10, error_spike_prob=0.15, latency_spike_prob=0.15)
            health = assess_health(metrics, service.health_thresholds)
            data.append({
                'service': service_name,
                'cpu_usage': metrics.cpu_usage,
                'memory_usage': metrics.memory_usage,
                'request_latency': metrics.request_latency,
                'error_rate': metrics.error_rate,
                'is_healthy': health.is_healthy
            })
    return pd.DataFrame(data)

print("Generating historical data for prediction model...")
historical_df = generate_historical_data(num_samples_per_service=500)
print(f"Generated {len(historical_df)} data points.")
print(historical_df.head())

X = historical_df[['cpu_usage', 'memory_usage', 'request_latency', 'error_rate']]
y = historical_df['is_healthy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training prediction model...")
model.fit(X_train, y_train)
print("Training complete.")

y_pred = model.predict(X_test)
print("\nPrediction Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred, target_names=['Unhealthy', 'Healthy']))

def predict_health(metrics):
    input_data = pd.DataFrame([{
        'cpu_usage': metrics.cpu_usage,
        'memory_usage': metrics.memory_usage,
        'request_latency': metrics.request_latency,
        'error_rate': metrics.error_rate
    }])
    prediction = model.predict(input_data)[0]
    return bool(prediction)

# --- 6. Healing Decision Engine ---

def decide_healing_action(service, prediction_is_healthy):
    if service.state == "running" and not prediction_is_healthy:
        print(f"Prediction: Service '{service.name}' might become unhealthy soon.")
        return None

    if service.state == "unhealthy":
        print(f"Service '{service.name}' is unhealthy ({service.current_health.status_message}).")
        if "latency" in service.current_health.status_message or "cpu" in service.current_health.status_message:
            print(f"Deciding to scale up '{service.name}' due to performance issues.")
            return "scale_up"
        else:
            print(f"Deciding to restart '{service.name}'.")
            return "restart"

    return None

# --- 7. Action Execution Engine ---

def execute_action(service, action):
    if action == "restart":
        print(f"Executing restart for '{service.name}'...")
        service.state = "restarting"
        time.sleep(random.uniform(1, 3))
        print(f"Restart of '{service.name}' complete. Assuming recovery.")
        service.base_metrics = ServiceMetrics(
            cpu_usage=max(5, service.base_metrics.cpu_usage - random.uniform(5, 10)),
            memory_usage=max(5, service.base_metrics.memory_usage - random.uniform(5, 10)),
            request_latency=max(10, service.base_metrics.request_latency - random.uniform(10, 20)),
            error_rate=max(0.1, service.base_metrics.error_rate - random.uniform(0.5, 1.5))
        )
        service.update_metrics()
        service.update_health()
        service.state = "running" if service.current_health.is_healthy else "unhealthy"

    elif action == "scale_up":
        print(f"Executing scale up for '{service.name}'...")
        service.state = "scaling_up"
        time.sleep(random.uniform(2, 5))
        print(f"Scale up of '{service.name}' complete.")
        service.base_metrics = ServiceMetrics(
            cpu_usage=max(5, service.base_metrics.cpu_usage * 0.7),
            memory_usage=max(5, service.base_metrics.memory_usage * 0.7),
            request_latency=max(10, service.base_metrics.request_latency * 0.8),
            error_rate=service.base_metrics.error_rate
        )
        service.update_metrics()
        service.update_health()
        service.state = "running" if service.current_health.is_healthy else "unhealthy"

# --- 8. Orchestrator Simulation Loop ---

def run_orchestrator(duration_seconds=30, check_interval_seconds=2):
    print(f"\n--- Starting Service Health Orchestrator for {duration_seconds} seconds ---")
    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        print(f"\n--- Time: {int(time.time() - start_time)}s ---")
        for name, service in services.items():
            print(f"Checking service: {name} (State: {service.state})")

            service.update_metrics()
            print(f"  Current Metrics: {service.current_metrics}")

            service.update_health()
            print(f"  Current Health: {service.current_health}")

            prediction_is_healthy = predict_health(service.current_metrics)
            print(f"  Predicted Healthy: {prediction_is_healthy}")

            action_to_take = None
            if service.state in ["running", "unhealthy"]:
                action_to_take = decide_healing_action(service, prediction_is_healthy)

            if action_to_take:
                execute_action(service, action_to_take)
            else:
                if service.state not in ["running", "unhealthy"]:
                    print(f"  Service '{service.name}' is in state '{service.state}'. No new action.")

        time.sleep(check_interval_seconds)

    print("\n--- Orchestrator Simulation Finished ---")

# --- 9. Run Simulation ---

print("Injecting simulated issue into Order Service...")
services['Order Service'].base_metrics = ServiceMetrics(
    cpu_usage=80,
    memory_usage=90,
    request_latency=300,
    error_rate=10
)

run_orchestrator(duration_seconds=40)
