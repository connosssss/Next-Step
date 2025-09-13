## Next Step


Next step is an app that uses sckit to train off of stock data and different ML models to predict future prices based purely on past data. The prediction program was written in python and uses flask, with the front end made using Next JS. For the different models itself, it uses both a linear regression and a random forest model to predict seperately, then checks the accuracy of both to make weights to then make a combined prediction.

### Installation
##### Backend
1. Enter the backend
``` cd backend```
2. Create and Activate the virtual Environment
```
python -m venv venv

To Activate
Windows:
venv\Scripts\activate
Mac/Linux:
venv/bin/activate

```
3. Install dependencies
```pip install -r requirements.txt```

##### Frontend
1. Enter Frontend(New Terminal)
``` cd frontend```
2. Install Node dependencies
```
npm install chart.js
npm install
```

### Running
In the backend terminal, run ``` python main.py``` and in the frontend terminal, run ```npm run dev```

### Images
<img width="1883" height="619" alt="image" src="https://github.com/user-attachments/assets/2fac856b-9b26-4a42-a693-5e699b0843a5" />
<img width="1768" height="962" alt="image" src="https://github.com/user-attachments/assets/14b53f1d-d503-458f-b39e-aa703039460e" />
<img width="1807" height="1063" alt="image" src="https://github.com/user-attachments/assets/3888c89f-7d71-4416-a38f-8c0c7067f576" />

