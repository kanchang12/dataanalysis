from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import numpy as np
import os
import openai
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objs as go
import json
from scipy import stats

load_dotenv()  # Load environment variables

app = Flask(__name__)

# Configure OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})
    
    # For GET requests, render the index.html template
    return render_template('index.html')

def load_data(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        return None
    except pd.errors.EmptyDataError:
        return pd.DataFrame()  # Return an empty DataFrame

def calculate_descriptive_stats(df):
    """Calculate descriptive statistics for numeric columns"""
    column_stats = {}
    
    for col in df.select_dtypes(include=['number']).columns:
        column_stats[col] = {
            'mean': float(df[col].mean()),
            'median': float(df[col].median()),
            'std': float(df[col].std()),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'skewness': float(stats.skew(df[col].dropna())),
            'kurtosis': float(stats.kurtosis(df[col].dropna()))
        }
    
    return column_stats

def generate_plotly_graph(df, graph_type, x_column, y_column=None):
    """
    Generate a Plotly graph based on the graph type and data.
    """
    # Set default height and width for better performance
    layout = go.Layout(
        height=400,
        width=600,
        autosize=True,
        margin=dict(l=50, r=50, b=50, t=50, pad=4)
    )

    try:
        if graph_type == 'scatter':
            # Sample data for large datasets to improve performance
            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size) if len(df) > sample_size else df

            fig = px.scatter(sample_df, x=x_column, y=y_column, title=f'{y_column} vs {x_column}')
            fig.update_layout(layout)

        elif graph_type == 'line':
            fig = px.line(df, x=x_column, y=y_column, title=f'{y_column} over {x_column}')
            fig.update_layout(layout)

        elif graph_type == 'bar':
            # Limit categories for better performance
            if df[x_column].nunique() > 20:
                top_categories = df[x_column].value_counts().nlargest(20).index
                filtered_df = df[df[x_column].isin(top_categories)]
            else:
                filtered_df = df

            fig = px.bar(filtered_df, x=x_column, y=y_column, title=f'{y_column} by {x_column}')
            fig.update_layout(layout)

        elif graph_type == 'histogram':
            fig = px.histogram(df, x=x_column, title=f'Distribution of {x_column}')
            fig.update_layout(layout)

        elif graph_type == 'box':
            fig = px.box(df, x=x_column, y=y_column, title=f'Box plot of {y_column} by {x_column}')
            fig.update_layout(layout)

        else:
            # Default to histogram of first numeric column if no valid graph type
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                fig = px.histogram(df, x=numeric_cols[0], title=f'Distribution of {numeric_cols[0]}')
                fig.update_layout(layout)
            else:
                return None

        # Simplify the data for faster loading
        fig.update_traces(marker=dict(opacity=0.7))

        return json.loads(fig.to_json())

    except Exception as e:
        print(f"Error generating {graph_type} graph: {str(e)}")  # Debug print
        return None

def generate_key_insights(df, numeric_cols):
    """Generate key statistical insights about the data"""
    insights = []
    
    # Basic dataset insights
    insights.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns.")
    
    # Check for any columns with missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        insights.append(f"Found missing values in columns: {', '.join(missing_cols)}.")
    
    # Add distribution insights for numeric columns
    for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
        if len(df[col].dropna()) > 0:
            # Calculate skewness and distribution type
            skewness = float(stats.skew(df[col].dropna()))
            if abs(skewness) < 0.5:
                insights.append(f"{col} shows a nearly symmetric distribution (skewness={skewness:.2f}).")
            elif skewness > 0.5:
                insights.append(f"{col} shows right-skewed distribution (skewness={skewness:.2f}).")
            else:
                insights.append(f"{col} shows left-skewed distribution (skewness={skewness:.2f}).")
            
            # Add z-score insights for outliers
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = int((z_scores > 3).sum())
            if outliers > 0:
                insights.append(f"{col} has {outliers} potential outliers (z-score > 3).")
    
    return insights[:5]  # Return no more than 5 insights

def get_column_index(df, index_str):
    """
    Safely convert a string index to an integer and validate it.
    Returns None if the index is invalid.
    """
    try:
        index = int(index_str)
        if 0 <= index < len(df.columns):
            return index
        else:
            print(f"Invalid column index: {index} (out of range)")
            return None
    except ValueError:
        print(f"Invalid column index: {index_str} (not an integer)")
        return None


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    filename = data.get('filename')
    df = load_data(filename)

    if df is None:
        return jsonify({'error': 'File not found'})

    try:
        # Calculate descriptive statistics
        column_stats = calculate_descriptive_stats(df)

        # Generate multiple visualizations
        visualizations = []
        generated_charts = set()
        possible_graph_types = ['histogram', 'scatter', 'bar', 'line', 'box']
        num_graphs = min(len(possible_graph_types), 3)

        # Show available columns with indices
        columns_with_indices = [f"{i}: {col}" for i, col in enumerate(df.columns)]
        
        for i in range(num_graphs):
            graph_type = possible_graph_types[i]

            prompt = f"""
            You are a data visualization expert. 
Please read the entire head or sample provided by the user, 
then think of the best questions users might be interested in.
This is not about how many rows, standard deviation, etc. 
Focus on questions that users can implement in their business or personal life directly,
like which product/category is performing better, which region is having more sales, etc.
Make sure to provide only those graphs which can help users fix their problems.
Don't suggest random or not useful graphs.

If the data has some features which are almost equal, no need to include them in the graph.
If weekly sales are constant, don't use them, but if monthly sales are significant, show that.
            Available columns (with indices):
            {columns_with_indices}

            Dataset Sample:
            {df.head().to_string()}
            Graph Type: {graph_type}

            RESPOND ONLY with column indices in this exact format:
            X-axis: [index number]
            Y-axis: [index number]

            Example correct response:
            X-axis: 0
            Y-axis: 3

            Only provide the two lines above, nothing else.
            For histogram, only X-axis is needed.
            Use only numeric indices, no text.
            """

            try:
                response = openai.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data visualization expert. Provide only numeric indices."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=50
                )

                ai_response = response.choices[0].message.content.strip()
                print(f"AI Response: {ai_response}")  # Debug print

                # Parse AI response for column indices
                x_index = None
                y_index = None

                for line in ai_response.split('\n'):
                    if 'x-axis:' in line.lower():
                        x_index = get_column_index(df, line.split(':')[1].strip())
                        print(f"Parsed x_index: {x_index}")  # Debug print
                    elif 'y-axis:' in line.lower():
                        y_index = get_column_index(df, line.split(':')[1].strip())
                        print(f"Parsed y_index: {y_index}")  # Debug print

                # Skip if no valid x_index
                if x_index is None:
                    print(f"Invalid x_index for {graph_type}")  # Debug print
                    continue

                # Convert indices to column names
                x_column = df.columns[x_index]
                y_column = df.columns[y_index] if y_index is not None else None

                print(f"Creating {graph_type} with x:{x_column} y:{y_column}")  # Debug print


                # Skip if no valid x_index
                if x_index is None:
                    print(f"Invalid x_index for {graph_type}")  # Debug print
                    continue

                # Convert indices to column names
                x_column = df.columns[x_index]
                y_column = df.columns[y_index] if y_index is not None else None

                print(f"Creating {graph_type} with x:{x_column} y:{y_column}")  # Debug print

                # Generate the graph
                plotly_graph = generate_plotly_graph(df, graph_type, x_column, y_column)
                if plotly_graph:
                    visualizations.append({
                        'type': graph_type,
                        'graph': json.dumps(plotly_graph),
                        'x_axis': x_column,
                        'y_axis': y_column
                    })
                    generated_charts.add((graph_type, x_column, y_column))

            except Exception as e:
                print(f"Error in graph generation: {str(e)}")  # Debug print
                continue

        # Generate key insights
        numeric_cols = df.select_dtypes(include=['number']).columns
        insights = generate_key_insights(df, numeric_cols)

        # Create summary statistics
        summary_stats = {
            'row_count': int(len(df)),
            'column_count': int(len(df.columns)),
            'numeric_columns': numeric_cols.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        }

        return Response(
            response=json.dumps({
                'insights': insights,
                'stats': summary_stats,
                'column_stats': column_stats,
                'graphs': visualizations,
                'success': True
            }, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        print(f"Analysis error: {str(e)}")  # Debug print
        return jsonify({'error': f"Error generating analysis: {e}"})



@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('question')
    filename = data.get('filename')
    df = load_data(filename)

    if df is None:
        return jsonify({'error': 'Data not loaded'})

    # Calculate some basic stats to append to the response
    numeric_cols = df.select_dtypes(include=['number']).columns
    stats_summary = ""
    
    if len(numeric_cols) > 0:
        sample_col = numeric_cols[0]
        mean_val = float(df[sample_col].mean())
        median_val = float(df[sample_col].median())
        std_val = float(df[sample_col].std())
        stats_summary = f"Key stats for {sample_col}: mean={mean_val:.2f}, median={median_val:.2f}, std={std_val:.2f}."

    # Use OpenAI to generate a response
    prompt = f"""
You are a data analyst. Analyze the data and provide insights based on the user's question.

User Question: {user_input}
Data Columns: {df.columns.tolist()}
Data Sample: {df.head().to_string()}

**Instructions:**

1.  **If the user's question requires statistical analysis or data insights:**
    * Provide EXACTLY 2-3 key points.
    * Each key point MUST be two to three SHORT sentence in NORMAL HUMAN LANGUAGE without any statistical terms.
    * Only if the data is relevant, provide stats data not any details following each key point, provide "Statistical Figures: [relevant statistical details]" in a separate line.

2.  **If the user's question is a simple greeting or does NOT require statistical analysis:**
    * Respond with a short, normal human language response WITHOUT any statistical figures or the "Key Point:" format.
    * Example: "Hello! How can I help you today?" or "I understand."
Make two line gaps between each line break to ensure good readability please
3.  DO NOT write paragraphs or lengthy explanations. Only provide the information as instructed above. Follow the format EXACTLY.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst. Provide concise statistical insights only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=70
        )
        analysis = response.choices[0].message.content.strip()
        
        # Ensure the analysis is concise (4-5 points)
        analysis_points = analysis.split('\n')
        if len(analysis_points) > 5:
            analysis = '\n'.join(analysis_points[:5])
        
        # Append stats summary if not present in the analysis
        if stats_summary and stats_summary not in analysis:
            analysis = f"{analysis}\n{stats_summary}"
            
        return jsonify({'analysis': analysis})
    except Exception as e:
        return jsonify({'error': f"Error generating analysis: {e}"})

if __name__ == '__main__':
    app.run(debug=True)