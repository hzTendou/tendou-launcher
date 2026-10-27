with open('C:/Users/Ali/.gemini/antigravity/brain/21634532-e372-4398-b3bb-c907e58069ae/.system_generated/tasks/task-485.log', 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        if '[HTTP-STREAM]' in line or 'Decode TPS Cold/Warm:' in line:
            print(line.strip())
