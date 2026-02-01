<script setup>
import { ref } from 'vue';
import axios from 'axios';

const Gender = ref('');
const Age = ref(null);
const Occupation = ref('');
const SleepDuration = ref(null);
const PhysicalActivityLevel = ref(null);
const StressLevel = ref(null);
const BMICategory = ref('');
const HeartRate = ref(null);
const DailySteps = ref(null);
const SleepDisorder = ref('');
const BloodPressureSystolic = ref(null);
const BloodPressureDiastolic = ref(null);

const classificationResult = ref(null);
const isLoading = ref(false);
const error = ref(null);

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const submitForm = async () => {
  isLoading.value = true;
  error.value = null;
  classificationResult.value = null;

  const payload = {
    Gender: Gender.value,
    Age: Age.value,
    Occupation: Occupation.value,
    SleepDuration: SleepDuration.value,
    PhysicalActivityLevel: PhysicalActivityLevel.value,
    StressLevel: StressLevel.value,
    BMICategory: BMICategory.value,
    HeartRate: HeartRate.value,
    DailySteps: DailySteps.value,
    SleepDisorder: SleepDisorder.value,
    BloodPressureSystolic: BloodPressureSystolic.value,
    BloodPressureDiastolic: BloodPressureDiastolic.value,
  };

  try {
    const response = await axios.post(API_URL, payload);
      
    classificationResult.value = response.data.NameQualitySleep;
      
  } catch (e) {
    if (e.response && e.response.data && e.response.data.detail) {
      error.value = e.response.data.detail[0].msg;
    } else {
      error.value = 'Connection error to API';
    }
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="form-container">
    <h1>How Well Do You Sleep?</h1>
    <h2>Which Are Your Habits/Style Of Life?</h2>
    
    <form @submit.prevent="submitForm">
      
      <div class="form-group">
        <label for="gender">What is your gender?:</label>
        <select id="gender" v-model="Gender" required>
          <option value="" disabled>Select</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div><br></br>
      
      <div class="form-group">
        <label for="age">How old are you?:</label>
        <input type="number" id="age" v-model.number="Age" required min="0">
      </div><br></br>

      <div class="form-group">
        <label for="occupation">What is your current occupation or job title?:</label>
        <select id="occupation" v-model="Occupation" required>
          <option value="" disabled>Select</option>
          <option value="Software Engineer">Software Engineer</option>
          <option value="Doctor">Doctor</option>
          <option value="Sales Representative">Sales Representative</option>
          <option value="Teacher">Teacher</option>
          <option value="Nurse">Nurse</option>
          <option value="Engineer">Engineer</option>
          <option value="Accountant">Accountant</option>
          <option value="Scientist">Scientist</option>
          <option value="Lawyer">Lawyer</option>
          <option value="Salesperson">Salesperson</option>
          <option value="Manager">Manager</option>
        </select>
      </div><br></br>
      
      <div class="form-group">
        <label for="sleepDuration">How many hours do you typically sleep per night?:</label>
        <input type="number" id="sleepDuration" v-model.number="SleepDuration" 
               required min="0" max="24" step="0.1">
      </div><br></br>

      <div class="form-group">
        <label for="physicalActivityLevel">On average, what is your daily physical activity level (in an activity score from 0-100)?:</label>
        <input type="number" id="physicalActivityLevel" v-model.number="PhysicalActivityLevel" 
               required min="0" max="100">
      </div><br></br>
      
      <div class="form-group">
        <label for="stressLevel">On a scale of 1 to 10 (1 being very calm, 10 being highly stressed), what is your average stress level?:</label>
        <input type="number" id="stressLevel" v-model.number="StressLevel" 
               required min="1" max="10">
      </div><br></br>

      <div class="form-group">
        <label for="bmiCategory">What is your current BMI Category?:</label>
        <select id="bmiCategory" v-model="BMICategory" required>
          <option value="" disabled>Select</option>
          <option value="Normal">Normal</option>
          <option value="Overweight">Overweight</option>
          <option value="Obese">Obese</option>
        </select>
      </div><br></br>

      <div class="form-group">
        <label for="heartRate">What is your average resting heart rate (in beats per minute)?:</label>
        <input type="number" id="heartRate" v-model.number="HeartRate" 
               required min="0" max="300">
      </div><br></br>
      
      <div class="form-group">
        <label for="dailySteps">Roughly, how many steps do you take on a typical day?:</label>
        <input type="number" id="dailySteps" v-model.number="DailySteps" 
               required min="0">
      </div><br></br>
      
      <div class="form-group">
        <label for="sleepDisorder">Do you currently have a diagnosed sleep disorder?:</label>
        <select id="sleepDisorder" v-model="SleepDisorder" required>
          <option value="" disabled>Select</option>
          <option value="No">No</option>
          <option value="Sleep Apnea">Sleep Apnea</option>
          <option value="Insomnia">Insomnia</option>
        </select>
      </div><br></br>

      <div class="form-group">
        <label for="bps">What is your Systolic Blood Pressure (the top number)?:</label>
        <input type="number" id="bps" v-model.number="BloodPressureSystolic" required min="0">
      </div><br></br>
      
      <div class="form-group">
        <label for="bpd">What is your Diastolic Blood Pressure (the bottom number)?:</label>
        <input type="number" id="bpd" v-model.number="BloodPressureDiastolic" required min="0">
      </div><br></br>

      <button type="submit" :disabled="isLoading">
        {{ isLoading ? 'Sending...' : 'Get prediction' }}
      </button>
    </form>
    
    <div v-if="isLoading" class="message loading">Loading...</div>
    
    <div v-if="error" class="message error">
      Error: {{ error }}
    </div>
    
    <div v-if="classificationResult" class="message success">
      Prediction Result: {{ classificationResult }}
    </div>
  </div>
</template>

<style scoped>
.form-container {
  max-width: 600px;
  margin: 50px auto;
  padding: 20px;
  border: 2.5px solid #B57EDC;
  background-color: white;
  border-radius: 8px;
}
.form-group {
  margin-bottom: 15px;
  align-items: center;
  display: flex;
  flex-direction: column;
}
label {
  display: block;
  width: 80%;
  margin-bottom: 5px;
  font-weight: bold;
  color: #5b2d79;
}
h1 , h2 {
  color: #B57EDC;
}
input , select {
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;

  width: 80%;
  padding: 8px;
  box-sizing: border-box;
  border: 2px solid #B57EDC;
  color: #5b2d79;
  text-align: center;
  font-size: normal;
}
button {
  background-color: #B57EDC;
  color: black;
  padding: 12px 20px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: normal;
  font-weight: bold;
}
button:hover {
  background-color: #E1D02A;
}
.message {
  margin-top: 20px;
  padding: 10px;
  border-radius: 4px;
}
.error {
  background-color: #fdd;
  color: #c00;
}
.success {
  background-color: #dfd;
  color: #0c0;
}
</style>