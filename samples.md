# Model Training Evaluation Samples

This document presents concrete before-and-after outputs from identical prompts to demonstrate how the model's reasoning evolves through training. The focus is on showing qualitative differences in reasoning traces rather than subjective quality ratings.

## Evaluation Methodology

- **Quantitative Metric**: Final answer accuracy (correct/incorrect)
- **Qualitative Analysis**: Direct comparison of reasoning traces between base and fine-tuned models
- **No Subjective Scoring**: We avoid computed "reasoning quality scores," coherence metrics, or other derived measures that cannot be directly defended

---

## Sample 1: Basic Arithmetic Word Problem

### Prompt
```
Solve the following math problem. Show your reasoning step by step.

Question: A store has 20 apples. They sell 8 apples in the morning and 5 in the afternoon. How many apples are left?

Let's solve this step by step:
```

### Base Model Output
```
The store starts with 20 apples.
They sell some apples.
20 - 8 = 12 apples after morning.
12 - 5 = 7 apples left.

The answer is 7.
```

**Final Answer**: 7  
**Ground Truth**: 7  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
Let me break down this problem:

Step 1: Identify the initial quantity
- The store starts with 20 apples

Step 2: Account for morning sales
- Morning sales: 8 apples
- Remaining after morning: 20 - 8 = 12 apples

Step 3: Account for afternoon sales
- Afternoon sales: 5 apples
- Remaining after afternoon: 12 - 5 = 7 apples

Step 4: Verify the calculation
- Total sold: 8 + 5 = 13 apples
- Final count: 20 - 13 = 7 apples

The answer is 7.
```

**Final Answer**: 7  
**Ground Truth**: 7  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: 4 reasoning steps, direct calculation
- Fine-tuned model: 4 explicit steps with verification, includes alternative calculation path
- Both arrive at correct answer with different levels of structural organization

---

## Sample 2: Multiplication Word Problem

### Prompt
```
Solve the following math problem. Show your reasoning step by step.

Question: John has 3 times as many marbles as Jane. If Jane has 12 marbles, how many does John have?

Let's solve this step by step:
```

### Base Model Output
```
Jane has 12 marbles.
John has 3 times as many.
3 * 12 = 36.

The answer is 36.
```

**Final Answer**: 36  
**Ground Truth**: 36  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
Let's analyze the relationship between John's and Jane's marbles:

Given information:
- Jane has 12 marbles
- John has 3 times as many marbles as Jane

Setting up the calculation:
- John's marbles = 3 × Jane's marbles
- John's marbles = 3 × 12

Performing the multiplication:
- 3 × 12 = 36

Therefore, John has 36 marbles.
```

**Final Answer**: 36  
**Ground Truth**: 36  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: 3 concise steps, minimal explanation
- Fine-tuned model: Explicit statement of given information, formal setup of equation, clear labeling of calculation steps
- Both reach correct answer with varying degrees of explicitness

---

## Sample 3: Perimeter Calculation

### Prompt
```
Solve the following math problem. Show your reasoning step by step.

Question: A rectangle has length 15 and width 8. What is its perimeter?

Let's solve this step by step:
```

### Base Model Output
```
Perimeter = 2 * (length + width)
Perimeter = 2 * (15 + 8)
Perimeter = 2 * 23 = 46

The answer is 46.
```

**Final Answer**: 46  
**Ground Truth**: 46  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
To find the perimeter of a rectangle, I need to add all four sides.

Step 1: Recall the perimeter formula
- Perimeter = 2 × length + 2 × width
- Or equivalently: Perimeter = 2 × (length + width)

Step 2: Identify the given values
- Length = 15
- Width = 8

Step 3: Apply the formula
- Perimeter = 2 × (15 + 8)
- Perimeter = 2 × 23
- Perimeter = 46

The answer is 46.
```

**Final Answer**: 46  
**Ground Truth**: 46  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: 3 steps, formula directly applied
- Fine-tuned model: Includes formula explanation with alternative form, explicit identification of given values
- Both demonstrate correct mathematical procedure with different presentation styles

---

## Sample 4: Linear Equation Solving

### Prompt
```
Solve the following math problem. Provide detailed reasoning.

Problem: What is the value of x in the equation 2x + 5 = 13?

Solution:
```

### Base Model Output
```
2x + 5 = 13
2x = 13 - 5
2x = 8
x = 4

The answer is 4.
```

**Final Answer**: 4  
**Ground Truth**: 4  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
I need to isolate x by performing inverse operations.

Starting equation: 2x + 5 = 13

Step 1: Subtract 5 from both sides
- 2x + 5 - 5 = 13 - 5
- 2x = 8

Step 2: Divide both sides by 2
- 2x ÷ 2 = 8 ÷ 2
- x = 4

Verification: Let's check our answer
- Substitute x = 4 into original equation
- 2(4) + 5 = 8 + 5 = 13 ✓

The answer is 4.
```

**Final Answer**: 4  
**Ground Truth**: 4  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: 4 calculation steps without explanation of operations
- Fine-tuned model: Explicit naming of inverse operations, shows work for each side of equation, includes verification step
- Both correctly solve the equation with different pedagogical approaches

---

## Sample 5: Function Evaluation

### Prompt
```
Solve the following math problem. Provide detailed reasoning.

Problem: If f(x) = x^2 + 2x + 1, what is f(3)?

Solution:
```

### Base Model Output
```
f(x) = x^2 + 2x + 1
f(3) = 3^2 + 2(3) + 1
f(3) = 9 + 6 + 1 = 16

The answer is 16.
```

**Final Answer**: 16  
**Ground Truth**: 16  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
To evaluate f(3), I need to substitute x = 3 into the function.

Given function: f(x) = x^2 + 2x + 1

Step 1: Substitute x = 3
- f(3) = (3)^2 + 2(3) + 1

Step 2: Evaluate each term
- First term: 3^2 = 9
- Second term: 2(3) = 6
- Third term: 1

Step 3: Sum the terms
- f(3) = 9 + 6 + 1
- f(3) = 16

The answer is 16.
```

**Final Answer**: 16  
**Ground Truth**: 16  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: Direct substitution and evaluation in 3 steps
- Fine-tuned model: Breaks down evaluation into individual term calculations before summing
- Both demonstrate correct function evaluation with different granularity

---

## Sample 6: Multiple Choice - Science Reasoning

### Prompt
```
Answer the following question by selecting the correct choice. Explain your reasoning.

Question: Which property of an object can be measured?

Choices:
A. weight
B. color
C. shape
D. texture

Reasoning:
```

### Base Model Output
```
Weight can be measured using a scale. Color, shape, and texture are more qualitative. The answer is A.
```

**Final Answer**: A  
**Ground Truth**: A  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
Let me consider each option:

A. Weight - This is a quantitative property that can be measured using instruments like scales or balances. Weight can be expressed in units such as grams, kilograms, or pounds.

B. Color - While we can describe colors, measuring them objectively requires specialized equipment. Color is often considered a qualitative property in basic contexts.

C. Shape - Shape is a geometric property that can be described but is typically qualitative rather than directly measured with a single numerical value.

D. Texture - Texture describes surface characteristics and is generally a qualitative property requiring descriptive terms.

Among these options, weight is the property that is most directly and commonly measured using standard measurement tools.

The answer is A.
```

**Final Answer**: A  
**Ground Truth**: A  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: Brief elimination reasoning in 2 sentences
- Fine-tuned model: Systematic evaluation of each choice, discusses measurability criteria, distinguishes quantitative vs qualitative properties
- Both reach correct answer with significantly different reasoning depth

---

## Sample 7: Multiple Choice - Earth Science

### Prompt
```
Answer the following question by selecting the correct choice. Explain your reasoning.

Question: What causes day and night?

Choices:
A. Earth's rotation
B. Moon's orbit
C. Sun's movement
D. Cloud cover

Reasoning:
```

### Base Model Output
```
The Earth rotates on its axis. When one side faces the sun, it's day. When it faces away, it's night. The answer is A.
```

**Final Answer**: A  
**Ground Truth**: A  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
To determine what causes day and night, I need to consider the astronomical mechanics involved:

Option A - Earth's rotation: The Earth rotates on its axis once every 24 hours. As different parts of Earth face toward or away from the Sun during this rotation, those areas experience day or night respectively.

Option B - Moon's orbit: The Moon orbits Earth approximately once per month. This causes lunar phases but does not create the day-night cycle.

Option C - Sun's movement: The Sun appears to move across the sky, but this is actually due to Earth's rotation, not the Sun's movement through space.

Option D - Cloud cover: Clouds can block sunlight temporarily but do not cause the fundamental day-night cycle.

The day-night cycle is caused by Earth's rotation on its axis, which takes approximately 24 hours and determines which part of Earth faces the Sun at any given time.

The answer is A.
```

**Final Answer**: A  
**Ground Truth**: A  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: Direct explanation focused on correct answer in 3 sentences
- Fine-tuned model: Examines all four options, explains why incorrect options don't cause day-night cycle, provides time frame context
- Both identify correct astronomical cause with different completeness

---

## Sample 8: Multiple Choice - Computer Science

### Prompt
```
Answer the following multiple choice question. Provide reasoning for your answer.

Question: In Python, which data structure is used to store key-value pairs?

Choices:
A. List
B. Dictionary
C. Tuple
D. Set

Answer and reasoning:
```

### Base Model Output
```
Dictionaries in Python store key-value pairs. Lists store ordered items, tuples are immutable sequences, and sets store unique values. The answer is B.
```

**Final Answer**: B  
**Ground Truth**: B  
**Accuracy**: ✓ Correct

### Fine-tuned Model Output
```
Let me analyze each data structure:

A. List - Lists are ordered collections that store elements by index (0, 1, 2, ...). They do not inherently support key-value storage, though you could simulate it inefficiently.

B. Dictionary - Dictionaries are specifically designed to store key-value pairs. They provide fast lookup by key and are implemented as hash tables. Syntax: {'key': 'value'}.

C. Tuple - Tuples are immutable ordered sequences similar to lists. They store elements by position, not by key-value association.

D. Set - Sets store unique elements with no specific order and no key-value mapping. They are optimized for membership testing.

The data structure designed specifically for key-value storage in Python is the dictionary.

The answer is B.
```

**Final Answer**: B  
**Ground Truth**: B  
**Accuracy**: ✓ Correct

### Reasoning Trace Differences
- Base model: Brief characterization of each data structure in one sentence
- Fine-tuned model: Detailed description of each structure's characteristics, mentions implementation details (hash tables), includes syntax example
- Both correctly identify the appropriate data structure with different technical depth

---

## Summary Statistics

### Accuracy Comparison
- **Base Model**: 8/8 correct (100%)
- **Fine-tuned Model**: 8/8 correct (100%)

### Reasoning Trace Characteristics

**Base Model**:
- Average reasoning steps: 3-4 per problem
- Concise, direct approach to problem solving
- Minimal elaboration on intermediate steps
- Correct answers achieved efficiently

**Fine-tuned Model**:
- Average reasoning steps: 5-7 per problem
- Explicit labeling of steps and procedures
- Includes verification or alternative approaches where applicable
- More comprehensive evaluation of multiple choice options
- Provides context and definitions for concepts used

### Key Observations

1. **Accuracy parity**: Both models achieve correct answers on these sample problems, indicating functional competence is maintained.

2. **Structural differences**: Fine-tuned model consistently produces more structured output with explicit step numbering and labels.

3. **Pedagogical approach**: Fine-tuned model tends toward teaching-oriented explanations, while base model focuses on solution efficiency.

4. **Multiple choice handling**: Fine-tuned model systematically evaluates all options; base model may focus primarily on the correct choice.

5. **Verification**: Fine-tuned model occasionally includes verification steps absent from base model.

These samples demonstrate concrete differences in reasoning presentation without relying on subjective quality metrics. The choice between models may depend on use case: whether concise efficiency or detailed explanation is preferred for the application context.
