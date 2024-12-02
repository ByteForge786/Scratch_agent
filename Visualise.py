class VisualizationTool(BaseTool):
    """Tool for determining and generating appropriate visualizations"""
    
    def __init__(self):
        self.viz_prompt_template = """
Given the following:

1. User Query: "{query}"
2. SQL Result Columns: {columns}
3. Number of Rows: {row_count}
4. Sample Data: {sample_data}

Determine if this data should be visualized and if so, what type of visualization would be most appropriate.

Consider:
1. Query intent (trend analysis, comparison, distribution, composition)
2. Data characteristics (number of dimensions, data types, cardinality)
3. Best practices for data visualization

Response format:
{{
    "needs_visualization": true/false,
    "explanation": "Reasoning for visualization decision",
    "visualization": {{
        "type": "line|bar|pie|scatter|histogram|heatmap|none",
        "config": {{
            "x_column": "column_name",
            "y_column": "column_name",
            "color_by": "column_name",  # optional
            "title": "Chart title",
            "aggregation": "sum|mean|count|none"  # optional
        }},
        "python_code": "Complete Python code using plotly.express to generate the visualization"
    }}
}}
"""

    def execute(self, query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Determine visualization needs and generate code"""
        try:
            sample_data = df.head(5).to_dict('records')
            prompt = self.viz_prompt_template.format(
                query=query,
                columns=list(df.columns),
                row_count=len(df),
                sample_data=sample_data
            )
            
            # Get LLM response for visualization decision
            viz_decision = self._call_llm(prompt)
            
            if viz_decision["needs_visualization"]:
                # Execute the generated Python code safely
                local_dict = {"df": df, "px": px}
                exec(viz_decision["visualization"]["python_code"], {}, local_dict)
                fig = local_dict.get("fig")
                
                return {
                    "status": "success",
                    "needs_visualization": True,
                    "explanation": viz_decision["explanation"],
                    "visualization": {
                        "config": viz_decision["visualization"]["config"],
                        "figure": fig
                    }
                }
            
            return {
                "status": "success",
                "needs_visualization": False,
                "explanation": viz_decision["explanation"]
            }
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {str(e)}")
            return {"status": "error", "error": str(e)}
