# LinkedIn Job Salary Data - Feature Dictionary

## Overview
- **Total Records**: 35,368 LinkedIn job postings
- **Total Features**: 62
- **Target Variables**: salary_yearly_min, salary_yearly_max, salary_yearly_med
- **Data Source**: LinkedIn job postings (merged with company, skills, benefits data)

---

## Feature Categories

### 1. Job Identifiers (2 features)
| Feature | Type | Description |
|---------|------|-------------|
| `job_id` | Integer | Unique identifier for each job posting |
| `salary_id` | Integer | Unique identifier for salary information |

### 2. Company Information (8 features)
| Feature | Type | Description |
|---------|------|-------------|
| `company_name` | Text | Name of the hiring company |
| `company_id` | Integer | Unique identifier for the company |
| `company_size` | Categorical | Size category (Small, Medium, Large, etc.) |
| `employee_count` | Integer | Number of employees at the company |
| `follower_count` | Integer | Number of LinkedIn followers for the company |
| `description_company` | Text | Company description and overview |
| `url` | Text | Company LinkedIn profile URL |
| `time_recorded` | Datetime | When company data was recorded |

### 3. Job Details (6 features)
| Feature | Type | Description |
|---------|------|-------------|
| `title` | Text | Job title (e.g., "Software Engineer", "Marketing Manager") |
| `description` | Text | Full job description including requirements and responsibilities |
| `formatted_work_type` | Categorical | Job type: Full-time, Part-time, Contract, Internship, etc. |
| `work_type` | Integer | Numeric encoding of work type |
| `formatted_experience_level` | Categorical | Entry level, Mid-Senior, Associate, Director, Executive |
| `skills_desc` | Text | Skills mentioned in the job description |

### 4. Location (8 features)
| Feature | Type | Description |
|---------|------|-------------|
| `location` | Text | Full location string (e.g., "San Francisco, CA") |
| `state` | Text | US state or region |
| `country` | Text | Country code or name |
| `city` | Text | City name |
| `zip_code` | Text | ZIP/postal code for job location |
| `zip_code_company` | Text | ZIP/postal code for company headquarters |
| `address` | Text | Full street address |
| `fips` | Text | Federal Information Processing Standards code |

### 5. Salary (Target Variables) (3 features)
| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `salary_yearly_min` | Float | Minimum annual salary in USD | $14,560 - $450,000 |
| `salary_yearly_max` | Float | Maximum annual salary in USD | $14,560 - $500,000 |
| `salary_yearly_med` | Float | Median annual salary in USD | Variable |

**Note**: All salaries have been converted to annual amounts:
- Hourly rates × 2,080 hours/year
- Monthly salaries × 12 months
- Weekly salaries × 52 weeks

### 6. Engagement Metrics (3 features)
| Feature | Type | Description |
|---------|------|-------------|
| `views` | Integer | Number of times the job posting was viewed |
| `applies` | Integer | Number of applications received |
| `sponsored` | Boolean | Whether the posting was a sponsored ad |

### 7. Posting Information (6 features)
| Feature | Type | Description |
|---------|------|-------------|
| `original_listed_time` | Datetime | When the job was originally posted |
| `listed_time` | Datetime | Current listing timestamp |
| `expiry` | Datetime | When the job posting expires |
| `closed_time` | Datetime | When the job was closed/filled |
| `posting_domain` | Text | Website domain where job was posted |
| `remote_allowed` | Boolean | Whether remote work is allowed (1.0 = Yes) |

### 8. Application Details (3 features)
| Feature | Type | Description |
|---------|------|-------------|
| `job_posting_url` | Text | URL to the full job posting |
| `application_url` | Text | URL to apply for the job |
| `application_type` | Categorical | Type of application process |

### 9. Industries (6 features)
| Feature | Type | Description |
|---------|------|-------------|
| `industry_count` | Integer | Number of industries associated with the job |
| `primary_industry` | Text | Main industry category |
| `all_industries` | Text | Comma-separated list of all relevant industries |
| `company_industry_count` | Integer | Number of industries the company operates in |
| `primary_company_industry` | Text | Company's primary industry |
| `all_company_industries` | Text | All industries the company is involved in |

### 10. Skills (3 features)
| Feature | Type | Description |
|---------|------|-------------|
| `skill_count` | Integer | Number of skills required for the job |
| `primary_skill` | Text | Most important skill required |
| `all_skills` | Text | Comma-separated list of all required skills |

### 11. Benefits (2 features)
| Feature | Type | Description |
|---------|------|-------------|
| `benefit_count` | Integer | Number of benefits offered |
| `benefits_list` | Text | Comma-separated list of all benefits |

### 12. Specialties (3 features)
| Feature | Type | Description |
|---------|------|-------------|
| `speciality_count` | Integer | Number of specialty areas |
| `primary_speciality` | Text | Main specialty or focus area |
| `all_specialities` | Text | All specialty areas related to the job |

### 13. Pay Period Information (8 features)
| Feature | Type | Description |
|---------|------|-------------|
| `pay_period` | Categorical | Original pay period from posting |
| `pay_period_salary` | Categorical | Pay period from salary table |
| `compensation_type` | Categorical | Type of compensation structure |
| `compensation_type_salary` | Categorical | Compensation type from salary table |
| `currency_salary` | Text | Currency code (mostly USD) |
| `max_salary_salary` | Float | Original max salary before conversion |
| `med_salary_salary` | Float | Original median salary before conversion |
| `min_salary_salary` | Float | Original min salary before conversion |

---

## Data Distribution

### Work Type
- Full-time: 81.0%
- Contract: 10.5%
- Part-time: 6.4%
- Other: 2.1%

### Experience Level
- Mid-Senior: 35.8%
- Entry: 25.5%
- Associate: 10.7%
- Director+: 5.6%

### Remote Work
- Remote allowed: 13.4%
- On-site: 86.6%

---

## Modeling Recommendations

### Primary Prediction Targets
1. **salary_yearly_med** - Best single target (most accurate when available)
2. **salary_yearly_max** - Upper bound prediction
3. **salary_yearly_min** - Lower bound prediction

### Key Predictive Features
Based on expected importance:
1. **Job Details**: title, description, formatted_experience_level
2. **Skills**: skill_count, all_skills, primary_skill
3. **Location**: state, city, location
4. **Company**: company_size, employee_count, follower_count
5. **Work Type**: formatted_work_type, remote_allowed
6. **Industries**: primary_industry, all_industries

### Features to Encode
- **Text features**: title, description, skills_desc, all_skills
- **Categorical**: formatted_work_type, formatted_experience_level, state, city
- **Boolean**: remote_allowed, sponsored

### Features to Consider Dropping
May be redundant or have low predictive value:
- Duplicate pay period columns (keep only one version)
- URLs (job_posting_url, application_url, url)
- Timestamps (unless modeling time trends)
- IDs (job_id, salary_id, company_id)

---

## Data Quality Notes

### Missing Values
- salary_yearly_med: Only 17.7% have values
- salary_yearly_min/max: 82.3% coverage
- Check individual features for missingness before modeling

### Outliers Removed
- Salaries < $10,000/year (likely errors)
- Salaries > $500,000/year (extreme outliers)
- Invalid ranges where min > max
- **Total removed**: 52 records

### Data Cleaning Steps Applied
1. Merged 11 original tables
2. Filtered for salary information
3. Converted all salaries to annual USD
4. Removed outliers and invalid entries
5. Dropped 4 redundant salary columns

---

## Usage Example

```python
import csv

# Load the clean data
with open('salary_data_final.csv', 'r') as f:
    reader = csv.DictReader(f)
    data = list(reader)

# Access target variable
for row in data[:5]:
    print(f"Job: {row['title']}")
    print(f"Salary Range: ${row['salary_yearly_min']} - ${row['salary_yearly_max']}")
    print(f"Location: {row['location']}")
    print(f"Experience: {row['formatted_experience_level']}")
    print()
```

---

**Last Updated**: 2025-11-03
**Dataset Version**: Final (62 features, 35,368 records)
