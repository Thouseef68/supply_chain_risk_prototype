import {
  BookOpen,
  BrainCircuit,
  Database,
  Upload,
  SlidersHorizontal,
  History,
  BarChart3,
  Users,
  ArrowRight,
  ShieldCheck
} from "lucide-react";


function Guide() {

  return (
    <div className="page-container">

      <div className="page-header">

        <div>

          <div className="eyebrow">
            SYSTEM GUIDE
          </div>

          <h1>
            How SupplyChainIQ Works
          </h1>

          <p>
            A simple guide to understanding the platform,
            its predictions, and how each part of the system
            can be used.
          </p>

        </div>

      </div>


      {/* INTRO */}

      <section className="guide-hero">

        <div className="guide-hero-icon">
          <BrainCircuit size={34} />
        </div>

        <div>

          <h2>
            Predict Risk. Understand Causes. Act Before Disruption.
          </h2>

          <p>
            SupplyChainIQ is a machine-learning based decision
            support prototype for analyzing supply-chain risk.
            It uses historical shipment information and trained
            CatBoost models to estimate the likelihood of
            late delivery or supply-chain disruption.
          </p>

        </div>

      </section>


      {/* WHAT DOES IT DO */}

      <section className="guide-section">

        <div className="guide-section-title">

          <BookOpen size={20} />

          <div>
            <h2>
              What does SupplyChainIQ do?
            </h2>

            <p>
              The platform turns shipment information into
              understandable risk predictions.
            </p>
          </div>

        </div>


        <div className="guide-cards">

          <div className="guide-card">

            <Database size={22} />

            <h3>
              Uses shipment data
            </h3>

            <p>
              The system works with operational shipment
              information such as shipping mode, location,
              market, product information, delivery schedule,
              distance, carrier reliability, weather and
              geopolitical conditions depending on the model.
            </p>

          </div>


          <div className="guide-card">

            <BrainCircuit size={22} />

            <h3>
              Applies machine learning
            </h3>

            <p>
              Trained CatBoost classification models analyze
              the supplied information and estimate the
              probability of the selected risk.
            </p>

          </div>


          <div className="guide-card">

            <BarChart3 size={22} />

            <h3>
              Makes the result understandable
            </h3>

            <p>
              Results are shown as risk levels, probabilities,
              analytics, model information and historical
              prediction records.
            </p>

          </div>

        </div>

      </section>


      {/* HOW IT WORKS */}

      <section className="guide-section">

        <div className="guide-section-title">

          <ArrowRight size={20} />

          <div>
            <h2>
              How does the system work?
            </h2>

            <p>
              The complete prediction flow can be understood
              in five simple steps.
            </p>
          </div>

        </div>


        <div className="workflow">

          <div className="workflow-step">

            <div className="workflow-number">
              01
            </div>

            <div>
              <h3>
                Provide shipment information
              </h3>

              <p>
                A user enters shipment or operational
                information manually, or uploads multiple
                records through Batch Analysis.
              </p>
            </div>

          </div>


          <div className="workflow-line" />


          <div className="workflow-step">

            <div className="workflow-number">
              02
            </div>

            <div>
              <h3>
                Identify the prediction type
              </h3>

              <p>
                SupplyChainIQ supports two prediction
                workflows: late-delivery risk and
                supply-chain disruption risk.
              </p>
            </div>

          </div>


          <div className="workflow-line" />


          <div className="workflow-step">

            <div className="workflow-number">
              03
            </div>

            <div>
              <h3>
                Run the machine-learning model
              </h3>

              <p>
                The appropriate trained CatBoost model
                processes the supplied features and produces
                a probability estimate.
              </p>
            </div>

          </div>


          <div className="workflow-line" />


          <div className="workflow-step">

            <div className="workflow-number">
              04
            </div>

            <div>
              <h3>
                Convert the result into a risk level
              </h3>

              <p>
                The predicted probability is presented as
                a practical risk level such as Low, Medium,
                or High.
              </p>
            </div>

          </div>


          <div className="workflow-line" />


          <div className="workflow-step">

            <div className="workflow-number">
              05
            </div>

            <div>
              <h3>
                Review and act
              </h3>

              <p>
                Users can review the result, compare
                scenarios, analyze patterns, and keep a
                history of previous assessments.
              </p>
            </div>

          </div>

        </div>

      </section>


      {/* TWO MODELS */}

      <section className="guide-section">

        <div className="guide-section-title">

          <BrainCircuit size={20} />

          <div>
            <h2>
              The two prediction models
            </h2>

            <p>
              Each model answers a different supply-chain question.
            </p>
          </div>

        </div>


        <div className="model-guide-grid">

          <div className="model-guide-card">

            <div className="model-guide-icon">
              <ShieldCheck size={23} />
            </div>

            <h3>
              DataCo CatBoost
            </h3>

            <span>
              Late Delivery Risk
            </span>

            <p>
              This model is based on the DataCo supply-chain
              dataset and estimates the risk that a shipment
              will experience late delivery.
            </p>

            <div className="guide-list">

              <div>
                ✓ Shipping information
              </div>

              <div>
                ✓ Customer and order information
              </div>

              <div>
                ✓ Product and sales information
              </div>

              <div>
                ✓ Geographic and market information
              </div>

              <div>
                ✓ Date and engineered interaction features
              </div>

            </div>

          </div>


          <div className="model-guide-card">

            <div className="model-guide-icon">
              <ShieldCheck size={23} />
            </div>

            <h3>
              Disruption CatBoost
            </h3>

            <span>
              Supply Chain Disruption Risk
            </span>

            <p>
              This model uses disruption-oriented operational
              information to estimate the probability of a
              supply-chain disruption.
            </p>

            <div className="guide-list">

              <div>
                ✓ Origin and destination
              </div>

              <div>
                ✓ Transport mode
              </div>

              <div>
                ✓ Distance and shipment weight
              </div>

              <div>
                ✓ Fuel price conditions
              </div>

              <div>
                ✓ Geopolitical and weather conditions
              </div>

              <div>
                ✓ Carrier reliability and lead time
              </div>

            </div>

          </div>

        </div>

      </section>


      {/* HOW TO USE */}

      <section className="guide-section">

        <div className="guide-section-title">

          <BookOpen size={20} />

          <div>
            <h2>
              How to use the website
            </h2>

            <p>
              A simple walkthrough for a first-time user.
            </p>
          </div>

        </div>


        <div className="usage-grid">

          <div className="usage-card">

            <div className="usage-icon">
              01
            </div>

            <h3>
              Start at Command Center
            </h3>

            <p>
              Get an overview of the system, model status,
              shipment analytics and prediction activity.
            </p>

          </div>


          <div className="usage-card">

            <div className="usage-icon">
              02
            </div>

            <h3>
              Make a Risk Prediction
            </h3>

            <p>
              Enter shipment information and run an individual
              prediction to see the estimated risk.
            </p>

          </div>


          <div className="usage-card">

            <div className="usage-icon">
              03
            </div>

            <h3>
              Run Batch Analysis
            </h3>

            <p>
              Upload a supported CSV file to analyze multiple
              shipments together. The current prototype supports
              up to 6,000 records per batch.
            </p>

          </div>


          <div className="usage-card">

            <div className="usage-icon">
              04
            </div>

            <h3>
              Explore Shipments
            </h3>

            <p>
              Search and filter shipment records to inspect
              individual operational data and associated risk.
            </p>

          </div>


          <div className="usage-card">

            <div className="usage-icon">
              05
            </div>

            <h3>
              Test What-If Scenarios
            </h3>

            <p>
              Change operational conditions and observe how
              the selected model responds to the scenario.
            </p>

          </div>


          <div className="usage-card">

            <div className="usage-icon">
              06
            </div>

            <h3>
              Review Prediction History
            </h3>

            <p>
              Previously generated predictions are stored locally
              so users can review past assessments.
            </p>

          </div>

        </div>

      </section>


      {/* WHO CAN USE */}

      <section className="guide-section">

        <div className="guide-section-title">

          <Users size={20} />

          <div>
            <h2>
              Who can use SupplyChainIQ?
            </h2>

            <p>
              The prototype is designed around practical
              supply-chain risk analysis.
            </p>
          </div>

        </div>


        <div className="user-grid">

          <div className="user-card">

            <h3>
              Supply Chain Managers
            </h3>

            <p>
              Can use risk predictions to identify shipments
              that may require closer monitoring.
            </p>

          </div>


          <div className="user-card">

            <h3>
              Logistics Teams
            </h3>

            <p>
              Can evaluate operational conditions such as
              transport mode, lead time, distance and carrier
              reliability.
            </p>

          </div>


          <div className="user-card">

            <h3>
              Procurement & Operations Teams
            </h3>

            <p>
              Can use scenario analysis to examine different
              operational conditions before making decisions.
            </p>

          </div>


          <div className="user-card">

            <h3>
              Data Analysts
            </h3>

            <p>
              Can use analytics, shipment exploration and
              model intelligence to investigate risk patterns.
            </p>

          </div>


          <div className="user-card">

            <h3>
              Researchers & Students
            </h3>

            <p>
              Can use the prototype to demonstrate data mining,
              feature engineering, classification and predictive
              analytics in a supply-chain context.
            </p>

          </div>

        </div>

      </section>


      {/* IMPORTANT LIMITATION */}

      <section className="guide-notice">

        <div>
          <ShieldCheck size={22} />
        </div>

        <div>

          <h3>
            Understanding the predictions
          </h3>

          <p>
            SupplyChainIQ is a predictive decision-support
            prototype. A risk probability represents the model's
            estimated likelihood based on the information supplied
            to it. It is not a guarantee that an event will or
            will not happen.
          </p>

          <p>
            What-If Analysis shows how the trained model responds
            to a changed scenario. It should not be interpreted
            as proof that changing one variable alone will cause
            the predicted outcome.
          </p>

        </div>

      </section>


      {/* ABOUT THIS PROJECT */}

      <section className="guide-section">

        <div className="guide-section-title">

          <BrainCircuit size={20} />

          <div>
            <h2>
              About this project
            </h2>

            <p>
              The purpose, scope and basis of the system.
            </p>
          </div>

        </div>


        <div className="about-project">

          <p>
            SupplyChainIQ is a machine-learning based
            supply-chain risk prediction prototype developed
            using data mining and predictive analytics
            techniques.
          </p>

          <p>
            The system analyzes shipment and operational
            information and uses trained CatBoost
            classification models to estimate two
            different types of risk: late-delivery risk and
            supply-chain disruption risk.
          </p>

          <p>
            The platform combines prediction, batch
            analysis, analytics, scenario simulation,
            shipment exploration and prediction history
            into a single web application.
          </p>

        </div>

      </section>


      {/* TECHNOLOGY USED */}

      <section className="guide-section">

        <div className="guide-section-title">

          <Database size={20} />

          <div>
            <h2>
              Technology used
            </h2>

            <p>
              The tools and datasets behind the platform.
            </p>
          </div>

        </div>


        <div className="tech-grid">

          <div className="tech-card">
            <h3>Frontend</h3>
            <div className="tech-list">
              <div>React</div>
              <div>Vite</div>
              <div>JavaScript</div>
              <div>Recharts</div>
              <div>Lucide React</div>
            </div>
          </div>

          <div className="tech-card">
            <h3>Backend</h3>
            <div className="tech-list">
              <div>Python</div>
              <div>FastAPI</div>
              <div>Pandas</div>
              <div>NumPy</div>
            </div>
          </div>

          <div className="tech-card">
            <h3>Machine Learning</h3>
            <div className="tech-list">
              <div>CatBoost</div>
              <div>Feature Engineering</div>
              <div>Classification</div>
            </div>
          </div>

          <div className="tech-card">
            <h3>Storage</h3>
            <div className="tech-list">
              <div>SQLite</div>
            </div>
          </div>

          <div className="tech-card wide">
            <h3>Data</h3>
            <div className="tech-list">
              <div>DataCo Supply Chain Dataset</div>
              <div>Supply Chain Disruption Dataset</div>
            </div>
          </div>

        </div>

      </section>

    </div>
  );
}

export default Guide;