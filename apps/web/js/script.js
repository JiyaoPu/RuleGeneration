// ----- Rule Section -----
// 1. Initial Population Charts
var initPopData = {
    labels: ["Type A", "Type B", "Type C"],
    values: [50, 30, 20]
  };
  
  // Pie Chart: Ratio
  Plotly.newPlot('pieChart', [{
    values: initPopData.values,
    labels: initPopData.labels,
    type: 'pie'
  }], { title: 'Initial Population Ratio' });
  
  // Histogram (Bar chart): Count
  Plotly.newPlot('histogramPopulation', [{
    x: initPopData.labels,
    y: initPopData.values,
    type: 'bar'
  }], { title: 'Initial Population Count' });
  
  // 2. Round & Trade Numbers Histogram
  var rounds = Array.from({length: 10}, (_, i) => i + 1);
  var tradeNumbers = [5, 7, 6, 8, 9, 4, 7, 8, 10, 6];
  
  Plotly.newPlot('histogramRoundsTrades', [{
    x: rounds,
    y: tradeNumbers,
    type: 'bar'
  }], {
    title: 'Round vs Trade Numbers',
    xaxis: { title: 'Round Number' },
    yaxis: { title: 'Trade Number' }
  });
  
  // 3. Reproduction Number Visualization
  var reproductionNumber = 15;
  var reproDiv = document.getElementById('reproductionDisplay');
  for (var i = 0; i < reproductionNumber; i++) {
    var icon = document.createElement('i');
    icon.className = 'fas fa-user person-icon';
    reproDiv.appendChild(icon);
  }
  
  // ----- Strategy Section -----
  // 示例策略趋势数据（散点图）
  var strategyData = [{
    x: [1, 2, 3, 4, 5],
    y: [10, 15, 13, 17, 16],
    type: 'scatter',
    mode: 'lines+markers',
    marker: { color: 'rgb(0, 123, 255)' }
  }];
  
  Plotly.newPlot('strategyChart', strategyData, {
    title: 'Strategy Trend Over Rounds',
    xaxis: { title: 'Round Number' },
    yaxis: { title: 'Strategy Metric' }
  });
  
  // ----- Evaluation Section -----
  // 示例评价数据：合作率、个体收入、基尼系数
  var evaluationData = [{
    x: ['Cooperation Rate', 'Individual Income', 'Gini Coefficient'],
    y: [0.85, 150, 0.3],
    type: 'bar',
    marker: { color: ['#28a745', '#007bff', '#ffc107'] }
  }];
  
  Plotly.newPlot('evaluationChart', evaluationData, {
    title: 'Evaluation Metrics',
    xaxis: { title: 'Metrics' },
    yaxis: { title: 'Value' }
  });
  

  