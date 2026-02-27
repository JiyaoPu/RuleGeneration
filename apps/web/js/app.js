// /js/app.js
document.addEventListener("DOMContentLoaded", function () {
  /*
    配置说明：
    外层采用 row 布局，将页面分为左右两部分：
    - 左侧 stack：Input 页面（5个标签页：RuleSetting, StrategySetting, EvaluationExpectation, DesignerEvaluator, AgentTraining）
    - 右侧 stack：Index 页面（3个标签页：ruleVisualization, strategyVisualization, evaluationVisualization）
  */

  // ============ helpers：从 /api/latest 读数据（Blob -> App Service -> Frontend） ============
  function safeJsonParse(x, fallback) {
    try {
      if (x == null) return fallback;
      if (typeof x === "string") return JSON.parse(x);
      return x;
    } catch (e) {
      return fallback;
    }
  }

  // 拉取 latest metrics.json（通过后端 /api/latest）
  async function fetchLatestMetrics() {
    const url = apiUrl("/latest");
    if (!url) return null;

    const res = await fetch(
      url + (url.includes("?") ? "&" : "?") + "ts=" + Date.now(),
      {
        method: "GET",
        headers: { Accept: "application/json", "Cache-Control": "no-cache" },
      }
    );

    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(`GET /latest failed: ${res.status} ${text}`.trim());
    }
    return await res.json();
  }

  // 从 metrics.json 抽取 epochs 数组（兼容 epochs 是数组 / 字符串）
  function extractEpochs(metrics) {
    if (!metrics) return [];
    let epochs = metrics.epochs;
    epochs = safeJsonParse(epochs, []);
    if (!Array.isArray(epochs)) return [];
    // 按 epoch 排序（确保稳定）
    epochs.sort((a, b) => (a.epoch ?? 0) - (b.epoch ?? 0));
    return epochs;
  }

  // ============ GoldenLayout config ============
  var config = {
    content: [
      {
        type: "row",
        content: [
          {
            type: "stack",
            width: 30,
            content: [
              {
                type: "component",
                componentName: "RuleSetting",
                title: "Rule Setting",
              },
              {
                type: "component",
                componentName: "StrategySetting",
                title: "Strategy Setting",
              },
              {
                type: "component",
                componentName: "EvaluationExpectation",
                title: "Evaluation Expectation",
              },
              {
                type: "component",
                componentName: "DesignerEvaluator",
                title: "Designer & Evaluator",
              },
              {
                type: "component",
                componentName: "AgentTraining",
                title: "Agent Training",
              },
            ],
          },
          {
            type: "stack",
            width: 50,
            content: [
              {
                type: "component",
                componentName: "ruleVisualization",
                title: "Rule Visualization",
              },
              {
                type: "component",
                componentName: "strategyVisualization",
                title: "Strategy Visualization",
              },
              {
                type: "component",
                componentName: "evaluationVisualization",
                title: "Evaluation Visualization",
              },
            ],
          },
        ],
      },
    ],
  };

  var myLayout = new GoldenLayout(
    config,
    document.getElementById("layoutContainer")
  );

  /********** 左侧区域组件注册 **********/
  // 1. RuleSetting（Rule Setting）
  myLayout.registerComponent("RuleSetting", function (container, state) {
    var html = `
      <div id="RuleSetting">
        <h2>Rule setting</h2>
        <div class="slider-section">
          <h3>Initial population</h3>
          <div id="sliderContainer"></div>
        </div>
        <div class="trade-rules-section">
          <h3>Trade Rules</h3>
          <table>
            <tbody>
              <tr>
                <td>
                  <span class="icon cheat">&#x2694;</span>
                  <span class="icon cheat">&#x2694;</span>
                </td>
                <td>
                  <span class="icon cheat">&#x2694;</span>
                  <span class="icon cooperate">&#x1F91D;</span>
                </td>
                <td>
                  <span class="icon cooperate">&#x1F91D;</span>
                  <span class="icon cooperate">&#x1F91D;</span>
                </td>
              </tr>
              <tr>
                <td>
                  <input type="number" class="score-input" value="0">
                  <input type="number" class="score-input" value="0">
                </td>
                <td>
                  <input type="number" class="score-input" value="3">
                  <input type="number" class="score-input" value="-1">
                </td>
                <td>
                  <input type="number" class="score-input" value="2">
                  <input type="number" class="score-input" value="2">
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <div class="extra-section">
          <h3>Additional Controls</h3>
          <div class="control-row" id="roundNumberControl">
            <label>Round Number:</label>
            <input type="number" id="roundNumberInput" value="10" min="1" max="20">
            <div id="roundNumberSlider"></div>
          </div>
          <div class="control-row" id="reproductionNumberControl">
            <label>Reproduction Number:</label>
            <input type="number" id="reproductionNumberInput" value="0" min="0" max="20">
            <div id="reproductionNumberSlider"></div>
          </div>
          <div class="control-row" id="mistakePossibilityControl">
            <label>Mistake Possibility:</label>
            <div class="percent-input">
              <input type="number" id="mistakePossibilityInput" value="5" min="0" max="100" step="1">
            </div>
            <div id="mistakePossibilityDial"></div>
          </div>

          <div class="control-row" id="fixedRuleControl">
            <label>Fixed Rule:</label>
            <div id="fixedRuleToggle"></div>
            <input type="hidden" id="hiddenFixedRule" value="True">
          </div>
        </div>
      </div>
    `;
    container.getElement().html(html);

    var sliderContainer = container.getElement().find("#sliderContainer")[0];
    var sliderParamsClassic = [
      { name: "Random", color: "#2A2A99", defaultVal: 0 },
      { name: "Cheater", color: "#0066ff", defaultVal: 0 },
      { name: "Cooperator", color: "#ff9900", defaultVal: 0 },
      { name: "Copycat", color: "#990099", defaultVal: 0 },
      { name: "Grudger", color: "#ff0000", defaultVal: 0 },
      { name: "Detective", color: "#ffff00", defaultVal: 1 },
      { name: "AI", color: "#999999", defaultVal: 1 },
      { name: "Human", color: "#009900", defaultVal: 0 },
    ];

    sliderParamsClassic.forEach(function (param) {
      var rowDiv = document.createElement("div");
      rowDiv.className = "slider-row";
      rowDiv.style.display = "grid";
      rowDiv.style.gridTemplateColumns = "70px 50px auto";
      rowDiv.style.alignItems = "center";
      rowDiv.style.columnGap = "10px";
      rowDiv.style.marginBottom = "8px";

      var label = document.createElement("label");
      label.textContent = param.name;

      var input = document.createElement("input");
      input.type = "number";
      input.min = 0;
      input.max = 25;
      input.value = param.defaultVal;

      var sliderDiv = document.createElement("div");

      rowDiv.appendChild(label);
      rowDiv.appendChild(input);
      rowDiv.appendChild(sliderDiv);
      sliderContainer.appendChild(rowDiv);

      var slider = new Nexus.Slider(sliderDiv, {
        size: [120, 15],
        mode: "absolute",
        min: 0,
        max: 25,
        step: 1,
        value: param.defaultVal,
        orientation: "horizontal",
      });
      slider.colorize("accent", param.color);

      slider.on("change", function (val) {
        input.value = val;
      });

      input.addEventListener("change", function () {
        var newVal = parseInt(input.value, 10);
        if (isNaN(newVal)) newVal = 0;
        if (newVal < 0) newVal = 0;
        if (newVal > 25) newVal = 25;
        slider.value = newVal;
      });
    });

    // Round Number
    var roundNumberSliderDiv = container
      .getElement()
      .find("#roundNumberSlider")[0];
    var roundNumberInput = container.getElement().find("#roundNumberInput")[0];
    var roundNumberSlider = new Nexus.Slider(roundNumberSliderDiv, {
      size: [120, 15],
      mode: "absolute",
      min: 1,
      max: 20,
      step: 1,
      value: 10,
      orientation: "horizontal",
    });
    roundNumberSlider.on("change", function (val) {
      roundNumberInput.value = val;
    });
    roundNumberInput.addEventListener("change", function () {
      var newVal = parseInt(roundNumberInput.value, 10);
      if (isNaN(newVal)) newVal = 1;
      if (newVal < 1) newVal = 1;
      if (newVal > 100) newVal = 100;
      roundNumberSlider.value = newVal;
    });

    // Reproduction Number
    var reproductionNumberSliderDiv = container
      .getElement()
      .find("#reproductionNumberSlider")[0];
    var reproductionNumberInput = container
      .getElement()
      .find("#reproductionNumberInput")[0];
    var reproductionNumberSlider = new Nexus.Slider(
      reproductionNumberSliderDiv,
      {
        size: [120, 15],
        mode: "absolute",
        min: 0,
        max: 20,
        step: 1,
        value: 0,
        orientation: "horizontal",
      }
    );
    reproductionNumberSlider.on("change", function (val) {
      reproductionNumberInput.value = val;
    });
    reproductionNumberInput.addEventListener("change", function () {
      var newVal = parseInt(reproductionNumberInput.value, 10);
      if (isNaN(newVal)) newVal = 0;
      if (newVal < 0) newVal = 0;
      if (newVal > 25) newVal = 25;
      reproductionNumberSlider.value = newVal;
    });

    // Mistake Possibility
    var mistakePossibilityDialDiv = container
      .getElement()
      .find("#mistakePossibilityDial")[0];
    var mistakePossibilityInput = container
      .getElement()
      .find("#mistakePossibilityInput")[0];
    var mistakePossibilityDial = new Nexus.Dial(mistakePossibilityDialDiv, {
      size: [60, 60],
      interaction: "radial",
      mode: "relative",
      min: 0,
      max: 100,
      step: 1,
      value: 5,
    });
    mistakePossibilityDial.on("change", function (val) {
      mistakePossibilityInput.value = Math.round(val);
    });
    mistakePossibilityInput.addEventListener("change", function () {
      var newVal = parseFloat(mistakePossibilityInput.value);
      if (isNaN(newVal)) newVal = 0;
      if (newVal < 0) newVal = 0;
      if (newVal > 100) newVal = 100;
      mistakePossibilityDial.value = newVal;
    });

    // Fixed Rule Toggle
    var fixedRuleToggleDiv = container.getElement().find("#fixedRuleToggle")[0];
    var fixedRuleToggle = new Nexus.Toggle(fixedRuleToggleDiv, {
      size: [40, 20],
      state: true,
    });
    fixedRuleToggle.on("change", function () {
      document.getElementById("hiddenFixedRule").value = fixedRuleToggle.state
        ? "True"
        : "False";
    });
  });

  // 2. StrategySetting（Strategy Setting）
  myLayout.registerComponent("StrategySetting", function (container, state) {
    var html = `
      <div id="StrategySetting"> 
        <h2>Strategy Setting</h2>
        <div class="select-group">
          <label>Human Player:</label>
          <div id="humanPlayerSelect"></div>
          <input type="hidden" id="hiddenHumanPlayer" value="False">
        </div>
        <div class="select-group">
          <label>AI Type:</label>
          <div id="aiTypeSelect"></div>
          <input type="hidden" id="hiddenAIType" value="Q">
        </div>
        <div id="strategyButtonContainer" style="
            margin-top: 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
          ">
          <div id="buttonHolder" style="margin-bottom: 10px;"></div>
          <div id="buttonHint" style="font-size: 16px; color: #0074D9;">Run Program</div>
        </div>
      </div>
    `;
    container.getElement().html(html);

    var humanPlayerStrategySetting = container
      .getElement()
      .find("#humanPlayerSelect")[0];
    var aiTypeStrategySetting = container.getElement().find("#aiTypeSelect")[0];

    var humanPlayerSelect = new Nexus.Select(humanPlayerStrategySetting, {
      size: [100, 30],
      options: ["False", "True"],
    });
    humanPlayerSelect.value = 0;
    humanPlayerSelect.on("change", function () {
      document.getElementById("hiddenHumanPlayer").value =
        humanPlayerSelect.value;
    });

    var aiTypeSelect = new Nexus.Select(aiTypeStrategySetting, {
      size: [100, 30],
      options: ["Q", "DQN"],
    });
    aiTypeSelect.value = 0;
    aiTypeSelect.on("change", function () {
      document.getElementById("hiddenAIType").value = aiTypeSelect.value;
    });

    var buttonHolder = container.getElement().find("#buttonHolder")[0];
    var strategyButton = new Nexus.Button(buttonHolder, {
      size: [120, 40],
      text: "Run Program",
    });

    strategyButton.once("change", function () {
      var hintElement = document.getElementById("buttonHint");
      if (hintElement)
        hintElement.innerText = "Program is running, Please wait.";

      var settingSelectors = [
        "#RuleSetting input",
        "#StrategySetting input",
        "#EvaluationExpectation input",
        "#DesignerEvaluator input",
        "#AgentTraining input",
      ];
      var inputs = document.querySelectorAll(settingSelectors.join(", "));
      var settings = {};
      inputs.forEach(function (input, index) {
        var key = input.id || "input_" + index;
        settings[key] = input.value;
      });

      // ✅ send settings to backend to persist in Blob (run/settings.json & run/settings.txt)
      const saveUrl = apiUrl("/settings");
      if (!saveUrl) {
        console.warn("STATIC_ONLY enabled: skip /settings");
        return;
      }

      fetch(saveUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      })
        .then(async (r) => {
          const data = await r.json().catch(() => ({}));
          if (!r.ok)
            throw new Error(
              `POST /settings failed: ${r.status} ${JSON.stringify(data)}`
            );
          console.log("Settings saved:", data);
          return data;
        })
        .catch((err) => console.error("Save settings error:", err));

      // 发送到后端（你后续会改成触发 AML）
      const runUrl = apiUrl("/run");
      if (!runUrl) {
        console.warn("STATIC_ONLY enabled: skip /run");
        return;
      }
      fetch(runUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(settings),
      })
        .then((r) => r.json())
        .then((data) => console.log("Run response:", data))
        .catch((err) => console.error("Run error:", err));
    });
  });

  // 3. EvaluationExpectation（Evaluation Expectation）
  myLayout.registerComponent(
    "EvaluationExpectation",
    function (container, state) {
      var html = `
      <div id="EvaluationExpectation">
        <h2>Evaluation Expectation</h2>
        <div class="evaluation-group">
          <div class="param-unit" id="unit_cooperationRate">
            <div class="dial" id="cooperationRateDial"></div>
            <div class="percent-input">
              <input type="number" id="cooperationRateInput" value="50" min="0" max="100" step="1">
            </div>
            <div class="toggle" id="cooperationRateToggle"></div>
            <div class="param-name">Cooperation Rate</div>
          </div>
          <div class="param-unit" id="unit_individualIncome">
            <div class="dial" id="individualIncomeDial"></div>
            <input type="number" id="individualIncomeInput" value="2" min="0" step="0.01">
            <div class="toggle" id="individualIncomeToggle"></div>
            <div class="param-name">Individual Income</div>
          </div>
          <div class="param-unit" id="unit_giniCoefficient">
            <div class="dial" id="giniCoefficientDial"></div>
            <div class="percent-input">
              <input type="number" id="giniCoefficientInput" value="50" min="0" max="100" step="1">
            </div>
            <div class="toggle" id="giniCoefficientToggle"></div>
            <div class="param-name">Gini Coefficient</div>
          </div>
        </div>
      </div>
    `;
      container.getElement().html(html);

      var cooperationRateDialDiv = container
        .getElement()
        .find("#cooperationRateDial")[0];
      var cooperationRateInput = container
        .getElement()
        .find("#cooperationRateInput")[0];
      var cooperationRateDial = new Nexus.Dial(cooperationRateDialDiv, {
        size: [75, 75],
        interaction: "radial",
        mode: "relative",
        min: 0,
        max: 100,
        step: 1,
        value: 50,
      });
      cooperationRateDial.on("change", function (val) {
        cooperationRateInput.value = Math.round(val);
      });
      cooperationRateInput.addEventListener("change", function () {
        var newVal = parseFloat(cooperationRateInput.value);
        if (isNaN(newVal)) newVal = 0;
        if (newVal < 0) newVal = 0;
        if (newVal > 100) newVal = 100;
        cooperationRateDial.value = newVal;
      });
      new Nexus.Toggle(
        container.getElement().find("#cooperationRateToggle")[0],
        { size: [40, 20], state: false }
      );

      var individualIncomeDialDiv = container
        .getElement()
        .find("#individualIncomeDial")[0];
      var individualIncomeInput = container
        .getElement()
        .find("#individualIncomeInput")[0];
      var individualIncomeDial = new Nexus.Dial(individualIncomeDialDiv, {
        size: [75, 75],
        interaction: "radial",
        mode: "relative",
        min: 0,
        max: 10,
        step: 0.01,
        value: 2,
      });
      individualIncomeDial.on("change", function (val) {
        individualIncomeInput.value = val.toFixed(2);
      });
      individualIncomeInput.addEventListener("change", function () {
        var newVal = parseFloat(individualIncomeInput.value);
        if (isNaN(newVal)) newVal = 0;
        if (newVal < 0) newVal = 0;
        if (newVal > 10) newVal = 10;
        individualIncomeDial.value = newVal;
      });
      new Nexus.Toggle(
        container.getElement().find("#individualIncomeToggle")[0],
        { size: [40, 20], state: false }
      );

      var giniCoefficientDialDiv = container
        .getElement()
        .find("#giniCoefficientDial")[0];
      var giniCoefficientInput = container
        .getElement()
        .find("#giniCoefficientInput")[0];
      var giniCoefficientDial = new Nexus.Dial(giniCoefficientDialDiv, {
        size: [75, 75],
        interaction: "radial",
        mode: "relative",
        min: 0,
        max: 100,
        step: 1,
        value: 50,
      });
      giniCoefficientDial.on("change", function (val) {
        giniCoefficientInput.value = Math.round(val);
      });
      giniCoefficientInput.addEventListener("change", function () {
        var newVal = parseFloat(giniCoefficientInput.value);
        if (isNaN(newVal)) newVal = 0;
        if (newVal < 0) newVal = 0;
        if (newVal > 1) newVal = 1;
        giniCoefficientDial.value = newVal;
      });
      new Nexus.Toggle(
        container.getElement().find("#giniCoefficientToggle")[0],
        { size: [40, 20], state: false }
      );
    }
  );

  // 4. DesignerEvaluator（Designer & Evaluator）
  myLayout.registerComponent("DesignerEvaluator", function (container, state) {
    var html = `
      <div id="DesignerEvaluator">
        <h2>Designer and Evaluator</h2>
        <div class="deParam-row" id="row_batch_size">
          <label>batch_size</label>
          <input type="number" id="de_input_batch_size" value="1" min="1" max="100" step="1">
          <div class="de-slider-container" id="de_slider_batch_size"></div>
        </div>
        <div class="deParam-row" id="row_lr">
          <label>lr</label>
          <input type="number" id="de_input_lr" value="0.01" min="0" max="1" step="0.001">
          <div class="de-slider-container" id="de_slider_lr"></div>
        </div>
        <div class="deParam-row" id="row_b1">
          <label>b1</label>
          <input type="number" id="de_input_b1" value="0.5" min="0" max="1" step="0.01">
          <div class="de-slider-container" id="de_slider_b1"></div>
        </div>
        <div class="deParam-row" id="row_b2">
          <label>b2</label>
          <input type="number" id="de_input_b2" value="0.999" min="0" max="1" step="0.001">
          <div class="de-slider-container" id="de_slider_b2"></div>
        </div>
        <div class="deParam-row" id="row_RuleDimension">
          <label>RuleDimension</label>
          <input type="number" id="de_input_RuleDimension" value="17" min="1" max="50" step="1">
          <div class="de-slider-container" id="de_slider_RuleDimension"></div>
        </div>
        <div class="deParam-row" id="row_DE_train_episode">
          <label>DE_train_episode</label>
          <input type="number" id="de_input_DE_train_episode" value="5" min="1" max="100" step="1">
          <div class="de-slider-container" id="de_slider_DE_train_episode"></div>
        </div>
        <div class="deParam-row" id="row_DE_test_episode">
          <label>DE_test_episode</label>
          <input type="number" id="de_input_DE_test_episode" value="1" min="1" max="100" step="1">
          <div class="de-slider-container" id="de_slider_DE_test_episode"></div>
        </div>
        <div class="deParam-row" id="row_evaluationSize">
          <label>evaluationSize</label>
          <input type="number" id="de_input_evaluationSize" value="1" min="1" max="10" step="1">
          <div class="de-slider-container" id="de_slider_evaluationSize"></div>
        </div>
        <div class="deParam-row" id="row_layersNum">
          <label>layersNum</label>
          <input type="number" id="de_input_layersNum" value="1" min="1" max="10" step="1">
          <div class="de-slider-container" id="de_slider_layersNum"></div>
        </div>
      </div>
    `;
    container.getElement().html(html);

    var deParams = [
      { id: "batch_size", defaultVal: 1, min: 1, max: 100, step: 1 },
      { id: "lr", defaultVal: 0.01, min: 0, max: 1, step: 0.001 },
      { id: "b1", defaultVal: 0.5, min: 0, max: 1, step: 0.01 },
      { id: "b2", defaultVal: 0.999, min: 0, max: 1, step: 0.001 },
      { id: "RuleDimension", defaultVal: 17, min: 1, max: 50, step: 1 },
      { id: "DE_train_episode", defaultVal: 5, min: 1, max: 100, step: 1 },
      { id: "DE_test_episode", defaultVal: 1, min: 1, max: 100, step: 1 },
      { id: "evaluationSize", defaultVal: 1, min: 1, max: 10, step: 1 },
      { id: "layersNum", defaultVal: 1, min: 1, max: 10, step: 1 },
    ];

    deParams.forEach(function (param) {
      var input = container.getElement().find("#de_input_" + param.id)[0];
      var sliderDiv = container.getElement().find("#de_slider_" + param.id)[0];
      var slider = new Nexus.Slider(sliderDiv, {
        size: [120, 20],
        mode: "absolute",
        min: param.min,
        max: param.max,
        step: param.step,
        value: param.defaultVal,
        orientation: "horizontal",
      });
      slider.on("change", function (val) {
        input.value = val;
      });
      input.addEventListener("change", function () {
        var newVal = parseFloat(input.value);
        if (isNaN(newVal)) newVal = param.min;
        if (newVal < param.min) newVal = param.min;
        if (newVal > param.max) newVal = param.max;
        slider.value = newVal;
      });
    });
  });

  // 5. AgentTraining（Agent Training）
  myLayout.registerComponent("AgentTraining", function (container, state) {
    var html = `
      <div id="AgentTraining">
        <h2>Agent Training</h2>
        <div class="agentParam-row" id="row_agent_train_epoch">
          <label>agent_train_epoch</label>
          <input type="number" id="input_agent_train_epoch" value="10" min="1" max="20000" step="1">
          <div class="agent-slider-container" id="slider_agent_train_epoch"></div>
        </div>

        <div class="agentParam-row" id="row_gamma">
          <label>gamma</label>
          <input type="number" id="input_gamma" value="0.99" min="0" max="1" step="0.01">
          <div class="agent-slider-container" id="slider_gamma"></div>
        </div>
        <div class="agentParam-row" id="row_epsilon">
          <label>epsilon</label>
          <input type="number" id="input_epsilon" value="1.0" min="0" max="1" step="0.01">
          <div class="agent-slider-container" id="slider_epsilon"></div>
        </div>
        <div class="agentParam-row" id="row_epsilon_decay">
          <label>epsilon_decay</label>
          <input type="number" id="input_epsilon_decay" value="0.999" min="0" max="1" step="0.001">
          <div class="agent-slider-container" id="slider_epsilon_decay"></div>
        </div>
        <div class="agentParam-row" id="row_epsilon_min">
          <label>epsilon_min</label>
          <input type="number" id="input_epsilon_min" value="0.1" min="0" max="1" step="0.01">
          <div class="agent-slider-container" id="slider_epsilon_min"></div>
        </div>
        <div class="agentParam-row" id="row_memory_size">
          <label>memory_size</label>
          <input type="number" id="input_memory_size" value="10000" min="1000" max="100000" step="1000">
          <div class="agent-slider-container" id="slider_memory_size"></div>
        </div>
        <div class="agentParam-row" id="row_target_update">
          <label>target_update</label>
          <input type="number" id="input_target_update" value="10" min="1" max="100" step="1">
          <div class="agent-slider-container" id="slider_target_update"></div>
        </div>
        <div class="agentParam-row" id="row_state_size">
          <label>state_size</label>
          <input type="number" id="input_state_size" value="20" min="5" max="100" step="1">
          <div class="agent-slider-container" id="slider_state_size"></div>
        </div>
      </div>
    `;
    container.getElement().html(html);

    var agentParams = [
      { id: "agent_train_epoch", defaultVal: 10, min: 1, max: 20000, step: 1 },
      { id: "gamma", defaultVal: 0.99, min: 0, max: 1, step: 0.01 },
      { id: "epsilon", defaultVal: 1.0, min: 0, max: 1, step: 0.01 },
      { id: "epsilon_decay", defaultVal: 0.999, min: 0, max: 1, step: 0.001 },
      { id: "epsilon_min", defaultVal: 0.1, min: 0, max: 1, step: 0.01 },
      {
        id: "memory_size",
        defaultVal: 10000,
        min: 1000,
        max: 100000,
        step: 1000,
      },
      { id: "target_update", defaultVal: 10, min: 1, max: 100, step: 1 },
      { id: "state_size", defaultVal: 20, min: 5, max: 100, step: 1 },
    ];

    agentParams.forEach(function (param) {
      var input = container.getElement().find("#input_" + param.id)[0];
      var sliderDiv = container.getElement().find("#slider_" + param.id)[0];
      var slider = new Nexus.Slider(sliderDiv, {
        size: [120, 20],
        mode: "absolute",
        min: param.min,
        max: param.max,
        step: param.step,
        value: param.defaultVal,
        orientation: "horizontal",
      });
      slider.on("change", function (val) {
        input.value = val;
      });
      input.addEventListener("change", function () {
        var newVal = parseFloat(input.value);
        if (isNaN(newVal)) newVal = param.min;
        if (newVal < param.min) newVal = param.min;
        if (newVal > param.max) newVal = param.max;
        slider.value = newVal;
      });
    });
  });

  /********** 右侧区域组件注册 **********/
  // 6. ruleVisualization（Rule Visualization）
  console.log("ruleVisualization component init");

  myLayout.registerComponent("ruleVisualization", function (container, state) {
    var html = `
      <div id="ruleVisualizationContent">
        <h2 class="section-title">Rule Visualization</h2>
        <div class="row">
          <div class="col-md-6">
            <div id="pieChart" style="width:100%; height:300px;"></div>
          </div>
          <div class="col-md-6">
            <div id="lineChartInitialPopulation" style="width:100%; height:300px;"></div>
          </div>
        </div>
        <div class="row" style="margin-top:20px;">
          <div class="col-md-6">
            <div id="tradeRuleChart" style="width:100%; height:300px;"></div>
          </div>
          <div class="col-md-6">
            <div id="lineChartCombined" style="width:100%; height:300px;"></div>
          </div>
        </div>
      </div>
    `;
    container.getElement().html(html);

    var roles = [
      "Random",
      "Cheater",
      "Cooperator",
      "Copycat",
      "Grudger",
      "Detective",
      "AI",
      "Human",
    ];
    var tradeRuleLabels = [
      "Rule 1",
      "Rule 2",
      "Rule 3",
      "Rule 4",
      "Rule 5",
      "Rule 6",
    ];

    var pieDiv = container.getElement().find("#pieChart")[0];
    Plotly.newPlot(
      pieDiv,
      [{ values: [1, 1, 1, 1, 1, 1, 1, 1], labels: roles, type: "pie" }],
      {
        title: "Initial Population Distribution",
      }
    );

    var lineChartDiv = container
      .getElement()
      .find("#lineChartInitialPopulation")[0];
    var layoutInitial = {
      title: "Initial Population Over Time",
      xaxis: { title: "Epoch" },
      yaxis: { title: "Population Count" },
    };
    var initPopData = roles.map((r) => ({
      x: [],
      y: [],
      mode: "lines+markers",
      name: r,
    }));
    Plotly.newPlot(lineChartDiv, initPopData, layoutInitial);

    var tradeRuleDiv = container.getElement().find("#tradeRuleChart")[0];
    var layoutTrade = {
      title: "Trade Rule Changes Over Time",
      xaxis: { title: "Epoch" },
      yaxis: { title: "Trade Rule Value" },
    };
    var tradeRuleData = tradeRuleLabels.map((r) => ({
      x: [],
      y: [],
      mode: "lines+markers",
      name: r,
    }));
    Plotly.newPlot(tradeRuleDiv, tradeRuleData, layoutTrade);

    var lineChartCombinedDiv = container
      .getElement()
      .find("#lineChartCombined")[0];
    var layoutCombined = {
      title: "Combined Metrics Over Time",
      xaxis: { title: "Epoch" },
      yaxis: { title: "Round / Reproduction", range: [0, 20] },
      yaxis2: {
        title: "Mistake Possibility",
        overlaying: "y",
        side: "right",
        range: [0, 1],
      },
    };
    var combinedData = [
      { x: [], y: [], mode: "lines+markers", name: "Round Number" },
      { x: [], y: [], mode: "lines+markers", name: "Reproduction Number" },
      {
        x: [],
        y: [],
        mode: "lines+markers",
        name: "Mistake Possibility",
        yaxis: "y2",
      },
    ];
    Plotly.newPlot(lineChartCombinedDiv, combinedData, layoutCombined);

    container.on("resize", function () {
      Plotly.Plots.resize(pieDiv);
      Plotly.Plots.resize(lineChartDiv);
      Plotly.Plots.resize(tradeRuleDiv);
      Plotly.Plots.resize(lineChartCombinedDiv);
    });

    async function updateChartsFromLatest() {
      try {
        const metrics = await fetchLatestMetrics();
        if (!metrics) return;

        const epochs = extractEpochs(metrics);
        if (epochs.length === 0) return;

        const latest = epochs[epochs.length - 1];

        const latestCounts = safeJsonParse(latest.initial_agent_counts, []);
        Plotly.react(
          pieDiv,
          [
            {
              values: latestCounts.length
                ? latestCounts
                : [1, 1, 1, 1, 1, 1, 1, 1],
              labels: roles,
              type: "pie",
            },
          ],
          { title: "Initial Population (Epoch " + (latest.epoch ?? "?") + ")" }
        );

        // init population over time
        const initPop = roles.map((r) => ({
          x: [],
          y: [],
          mode: "lines+markers",
          name: r,
        }));
        epochs.forEach((rec) => {
          const ep = rec.epoch;
          const counts = safeJsonParse(rec.initial_agent_counts, []);
          for (let i = 0; i < roles.length; i++) {
            initPop[i].x.push(ep);
            initPop[i].y.push(counts[i] ?? null);
          }
        });
        Plotly.react(lineChartDiv, initPop, layoutInitial);

        // trade rules over time
        const tr = tradeRuleLabels.map((r) => ({
          x: [],
          y: [],
          mode: "lines+markers",
          name: r,
        }));
        epochs.forEach((rec) => {
          const ep = rec.epoch;
          const tradeArr = safeJsonParse(rec.trade_rules, []);
          for (let i = 0; i < tradeRuleLabels.length; i++) {
            tr[i].x.push(ep);
            tr[i].y.push(tradeArr[i] ?? null);
          }
        });
        Plotly.react(tradeRuleDiv, tr, layoutTrade);

        // combined metrics over time
        const comb = [
          { x: [], y: [], mode: "lines+markers", name: "Round Number" },
          { x: [], y: [], mode: "lines+markers", name: "Reproduction Number" },
          {
            x: [],
            y: [],
            mode: "lines+markers",
            name: "Mistake Possibility",
            yaxis: "y2",
          },
        ];
        epochs.forEach((rec) => {
          const ep = rec.epoch;
          comb[0].x.push(ep);
          comb[0].y.push(rec.round_number ?? null);
          comb[1].x.push(ep);
          comb[1].y.push(rec.reproduction_number ?? null);
          comb[2].x.push(ep);
          comb[2].y.push(rec.mistake_possibility ?? null);
        });
        Plotly.react(lineChartCombinedDiv, comb, layoutCombined);
      } catch (e) {
        console.error("updateChartsFromLatest error:", e);
      }
    }

    // init + poll（后续你用 Event Grid 推送时，把轮询换成“收到通知后调用 updateChartsFromLatest”）
    updateChartsFromLatest();
    const timerId = setInterval(updateChartsFromLatest, 5000);
    container.on("destroy", function () {
      clearInterval(timerId);
    });
  });

  // 7. strategyVisualization (Strategy Visualization)
  myLayout.registerComponent(
    "strategyVisualization",
    function (container, state) {
      var html = `
      <div id="strategyVisualizationContent">
        <h2 class="section-title">Strategy Visualization</h2>
        <div class="row">
          <div class="col-md-4" style="margin-bottom:20px;">
            <img src="/image/NPC_strategies.png" alt="NPC Strategy" style="width:58%; height:auto;">
            <img src="/image/human_strategy.png" alt="Human Strategy" style="width:35%; height:auto;">
          </div>
          <div class="col-md-4" style="margin-bottom:20px;">
            <img id="aiStrategyImg"
                src="/image/AI_strategies.png"
                alt="AI Strategy"
                style="width:100%; height:auto;">
          </div>
        </div>
      </div>
    `;
      container.getElement().html(html);

      // 仍然通过后端拿（后端从 Blob 读 latest/q_table_heatmap.png）
      function updateAIImage() {
        var img = container.getElement().find("#aiStrategyImg")[0];
        if (!img) return;

        const imgBase = apiUrl("/q_table_heatmap.png");
        if (!imgBase) {
          console.warn("STATIC_ONLY enabled: keep default AI_strategies.png");
          return;
        }
        img.src =
          imgBase + (imgBase.includes("?") ? "&" : "?") + "ts=" + Date.now();
      }

      updateAIImage();
      const timerId = setInterval(updateAIImage, 5000);
      container.on("destroy", function () {
        clearInterval(timerId);
      });
    }
  );

  /********** 8. evaluationVisualization（Evaluation Visualization） **********/
  myLayout.registerComponent(
    "evaluationVisualization",
    function (container, state) {
      var html = `
      <div id="evaluationVisualizationContent">
        <h3 class="section-title">Evaluation Visualization</h3>
        <div class="row" style="margin-bottom:20px;">
          <div class="col-md-6">
            <div id="cooperationRatePie" style="width:100%; height:300px;"></div>
          </div>
          <div class="col-md-6">
            <div id="cooperationRateLine" style="width:100%; height:300px;"></div>
          </div>
        </div>
        <div class="row" style="margin-bottom:20px;">
          <div class="col-md-6">
            <div id="individualIncomeHistogram" style="width:100%; height:300px;"></div>
          </div>
          <div class="col-md-6">
            <div id="individualIncomeLine" style="width:100%; height:300px;"></div>
          </div>
        </div>
        <div class="row">
          <div class="col-md-6">
            <div id="giniCoefficientRadar" style="width:100%; height:300px;"></div>
          </div>
          <div class="col-md-6">
            <div id="giniCoefficientLine" style="width:100%; height:300px;"></div>
          </div>
        </div>
      </div>
    `;
      container.getElement().html(html);

      // 9 类别：前8 + Overall（注意你原来 human 小写，我这里修正成 Human 以统一）
      var categories = [
        "Random",
        "Cheater",
        "Cooperator",
        "Copycat",
        "Grudger",
        "Detective",
        "AI",
        "Human",
        "Overall",
      ];

      var cooperationRatePieDiv = container
        .getElement()
        .find("#cooperationRatePie")[0];
      Plotly.newPlot(
        cooperationRatePieDiv,
        [
          {
            values: [1, 1, 1, 1, 1, 1, 1, 1, 1],
            labels: categories,
            type: "pie",
          },
        ],
        {
          title: "Final Cooperation Rate",
        }
      );

      var cooperationRateLineDiv = container
        .getElement()
        .find("#cooperationRateLine")[0];
      var layoutCoopLine = {
        title: "Cooperation Rate Evolution",
        xaxis: { title: "Epoch" },
        yaxis: { title: "Cooperation Rate", range: [0, 1] },
      };
      Plotly.newPlot(
        cooperationRateLineDiv,
        categories.map((c) => ({
          x: [],
          y: [],
          mode: "lines+markers",
          name: c,
        })),
        layoutCoopLine
      );

      var individualIncomeHistogramDiv = container
        .getElement()
        .find("#individualIncomeHistogram")[0];
      Plotly.newPlot(
        individualIncomeHistogramDiv,
        [{ x: categories, y: [0, 0, 0, 0, 0, 0, 0, 0, 0], type: "bar" }],
        {
          title: "Final Individual Income",
          xaxis: { title: "Category" },
          yaxis: { title: "Income" },
        }
      );

      var individualIncomeLineDiv = container
        .getElement()
        .find("#individualIncomeLine")[0];
      var layoutIncomeLine = {
        title: "Individual Income Evolution",
        xaxis: { title: "Epoch" },
        yaxis: { title: "Income" },
      };
      Plotly.newPlot(
        individualIncomeLineDiv,
        categories.map((c) => ({
          x: [],
          y: [],
          mode: "lines+markers",
          name: c,
        })),
        layoutIncomeLine
      );

      var giniCoefficientRadarDiv = container
        .getElement()
        .find("#giniCoefficientRadar")[0];
      Plotly.newPlot(
        giniCoefficientRadarDiv,
        [
          {
            type: "scatterpolar",
            r: [0, 0, 0, 0, 0, 0, 0, 0, 0],
            theta: categories,
            fill: "toself",
            name: "Final Gini Coefficient",
          },
        ],
        {
          polar: { radialaxis: { visible: true, range: [0, 1] } },
          showlegend: false,
          title: "Final Gini Coefficient",
        }
      );

      var giniCoefficientLineDiv = container
        .getElement()
        .find("#giniCoefficientLine")[0];
      var layoutGiniLine = {
        title: "Gini Coefficient Evolution",
        xaxis: { title: "Epoch" },
        yaxis: { title: "Gini Coefficient", range: [0, 1] },
      };
      Plotly.newPlot(
        giniCoefficientLineDiv,
        categories.map((c) => ({
          x: [],
          y: [],
          mode: "lines+markers",
          name: c,
        })),
        layoutGiniLine
      );

      container.on("resize", function () {
        Plotly.Plots.resize(cooperationRatePieDiv);
        Plotly.Plots.resize(cooperationRateLineDiv);
        Plotly.Plots.resize(individualIncomeHistogramDiv);
        Plotly.Plots.resize(individualIncomeLineDiv);
        Plotly.Plots.resize(giniCoefficientRadarDiv);
        Plotly.Plots.resize(giniCoefficientLineDiv);
      });

      async function updateEvaluationFromLatest() {
        try {
          const metrics = await fetchLatestMetrics();
          if (!metrics) return;

          const epochs = extractEpochs(metrics);
          if (epochs.length === 0) return;

          const latest = epochs[epochs.length - 1];

          const latest_coop = safeJsonParse(latest.cooperation_rate, []);
          const latest_income = safeJsonParse(latest.individual_income, []);
          const latest_gini = safeJsonParse(latest.gini_coefficient, []);

          // Pie (latest coop)
          Plotly.react(
            cooperationRatePieDiv,
            [
              {
                values: latest_coop.length
                  ? latest_coop
                  : new Array(categories.length).fill(0),
                labels: categories,
                type: "pie",
              },
            ],
            {
              title:
                "Final Cooperation Rate (Epoch " + (latest.epoch ?? "?") + ")",
            }
          );

          // Income histogram (latest)
          Plotly.react(
            individualIncomeHistogramDiv,
            [
              {
                x: categories,
                y: latest_income.length
                  ? latest_income
                  : new Array(categories.length).fill(0),
                type: "bar",
              },
            ],
            {
              title:
                "Final Individual Income (Epoch " + (latest.epoch ?? "?") + ")",
              xaxis: { title: "Category" },
              yaxis: { title: "Income" },
            }
          );

          // Radar (latest gini)
          Plotly.react(
            giniCoefficientRadarDiv,
            [
              {
                type: "scatterpolar",
                r: latest_gini.length
                  ? latest_gini
                  : new Array(categories.length).fill(0),
                theta: categories,
                fill: "toself",
                name: "Final Gini Coefficient",
              },
            ],
            {
              polar: { radialaxis: { visible: true, range: [0, 1] } },
              showlegend: false,
              title:
                "Final Gini Coefficient (Epoch " + (latest.epoch ?? "?") + ")",
            }
          );

          // Line charts over epochs
          const coopLineData = categories.map((c) => ({
            x: [],
            y: [],
            mode: "lines+markers",
            name: c,
          }));
          const incomeLineData = categories.map((c) => ({
            x: [],
            y: [],
            mode: "lines+markers",
            name: c,
          }));
          const giniLineData = categories.map((c) => ({
            x: [],
            y: [],
            mode: "lines+markers",
            name: c,
          }));

          epochs.forEach((rec) => {
            const ep = rec.epoch;
            const coop_vals = safeJsonParse(rec.cooperation_rate, []);
            const income_vals = safeJsonParse(rec.individual_income, []);
            const gini_vals = safeJsonParse(rec.gini_coefficient, []);
            for (let i = 0; i < categories.length; i++) {
              coopLineData[i].x.push(ep);
              coopLineData[i].y.push(coop_vals[i] ?? null);

              incomeLineData[i].x.push(ep);
              incomeLineData[i].y.push(income_vals[i] ?? null);

              giniLineData[i].x.push(ep);
              giniLineData[i].y.push(gini_vals[i] ?? null);
            }
          });

          Plotly.react(cooperationRateLineDiv, coopLineData, layoutCoopLine);
          Plotly.react(
            individualIncomeLineDiv,
            incomeLineData,
            layoutIncomeLine
          );
          Plotly.react(giniCoefficientLineDiv, giniLineData, layoutGiniLine);
        } catch (e) {
          console.error("updateEvaluationFromLatest error:", e);
        }
      }

      updateEvaluationFromLatest();
      const timerId = setInterval(updateEvaluationFromLatest, 5000);
      container.on("destroy", function () {
        clearInterval(timerId);
      });
    }
  );

  myLayout.init();
});
