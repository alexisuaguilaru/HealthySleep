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
        const response = await axios.post('http://localhost:8000/Classify', payload);
        
        classificationResult.value = response.data.NameQualitySleep;
        console.log(classificationResult.value);
        
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
    <h2>Which are your habits/style of life?</h2>
    
    <form @submit.prevent="submitForm">
      
      <div class="form-group">
        <label for="gender">Gender:</label>
        <select id="gender" v-model="Gender" required>
          <option value="" disabled>Seleccione</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>
      
      <div class="form-group">
        <label for="age">Age (years):</label>
        <input type="number" id="age" v-model.number="Age" required min="0">
      </div>

      <div class="form-group">
        <label for="occupation">Occupation:</label>
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
      </div>
      
      <div class="form-group">
        <label for="sleepDuration">Sleep Duration (0-24h):</label>
        <input type="number" id="sleepDuration" v-model.number="SleepDuration" 
               required min="0" max="24" step="0.1">
      </div>

      <div class="form-group">
        <label for="physicalActivityLevel">Physical Activity Level (0-100):</label>
        <input type="number" id="physicalActivityLevel" v-model.number="PhysicalActivityLevel" 
               required min="0" max="100">
      </div>
      
      <div class="form-group">
        <label for="stressLevel">Stress Level (1-10):</label>
        <input type="number" id="stressLevel" v-model.number="StressLevel" 
               required min="1" max="10">
      </div>

      <div class="form-group">
        <label for="bmiCategory">BMI Category:</label>
        <select id="bmiCategory" v-model="BMICategory" required>
          <option value="" disabled>Select</option>
          <option value="Normal">Normal</option>
          <option value="Overweight">Overweight</option>
          <option value="Obese">Obese</option>
        </select>
      </div>

      <div class="form-group">
        <label for="heartRate">Heart Rate (BPM):</label>
        <input type="number" id="heartRate" v-model.number="HeartRate" 
               required min="0" max="300">
      </div>
      
      <div class="form-group">
        <label for="dailySteps">Daily Steps:</label>
        <input type="number" id="dailySteps" v-model.number="DailySteps" 
               required min="0">
      </div>
      
      <div class="form-group">
        <label for="sleepDisorder">Sleep Disorder:</label>
        <select id="sleepDisorder" v-model="SleepDisorder" required>
          <option value="" disabled>Select</option>
          <option value="No">No</option>
          <option value="Sleep Apnea">Sleep Apnea</option>
          <option value="Insomnia">Insomnia</option>
        </select>
      </div>

      <div class="form-group">
        <label for="bps">Systolic Blood Pressure:</label>
        <input type="number" id="bps" v-model.number="BloodPressureSystolic" required min="0">
      </div>
      
      <div class="form-group">
        <label for="bpd">Diastolic Blood Pressure:</label>
        <input type="number" id="bpd" v-model.number="BloodPressureDiastolic" required min="0">
      </div>

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
  border: 1px solid #ccc;
  border-radius: 8px;
}
.form-group {
  margin-bottom: 15px;
}
label {
  display: block;
  margin-bottom: 5px;
  font-weight: bold;
}
input[type="number"], select {
  width: 100%;
  padding: 8px;
  box-sizing: border-box;
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