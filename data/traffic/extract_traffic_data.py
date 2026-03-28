
import pandas as pd
from bs4 import BeautifulSoup
import os


TRAFFIC_COLUMNS = [
	"Heavy Truck", "Medium Truck", "Small Truck", "Large Bus", "Medium Bus", "Micro Bus", "Utility", "Car", "Auto Rickshaw", "Motor Cycle", "Bi-Cycle", "Cycle Rickshaw", "Cart", "Motorized", "Non Motorized", "Total AADT", "(AADT)"
]


def parse_traffic_html(filepath):
	with open(filepath, encoding="ISO-8859-1") as f:
		soup = BeautifulSoup(f, "html.parser")
	# Find the main table with traffic data
	table = soup.find_all("table", attrs={"style": lambda x: x and "width:2500px;" in x})
	if not table:
		raise ValueError(f"Traffic table not found in {filepath}")
	table = table[0]
	rows = table.find_all("tr")
	data_rows = []
	def safe_float(val):
		val = val.strip().replace(",", "").replace("-", "0")
		if val.upper() == 'NS' or val == '':
			return 0.0
		try:
			return float(val)
		except Exception:
			return 0.0

	for row in rows:
		tds = row.find_all("td", class_="tdRow")
		if len(tds) >= 26:
			# Extract relevant fields
			link_start = tds[2].get_text(strip=True)
			link_end = tds[5].get_text(strip=True)
			chainage_start = tds[4].get_text(strip=True).replace("\xa0", "").replace(",", "").replace("-", "0")
			chainage_end = tds[7].get_text(strip=True).replace("\xa0", "").replace(",", "").replace("-", "0")
			try:
				chainage_start = float(chainage_start) if chainage_start else 0.0
			except Exception:
				chainage_start = 0.0
			try:
				chainage_end = float(chainage_end) if chainage_end else 0.0
			except Exception:
				chainage_end = 0.0
			# Traffic columns are from index 9 to 25 (inclusive)
			traffic = [safe_float(tds[i].get_text()) for i in range(9, 26)]
			data_rows.append({
				"link_start": link_start,
				"chainage_start": chainage_start,
				"link_end": link_end,
				"chainage_end": chainage_end,
				**{col: val for col, val in zip(TRAFFIC_COLUMNS, traffic)},
				"_link_id": tds[0].get_text(strip=True),
				"_desc": tds[1].get_text(strip=True)
			})
	return data_rows


def merge_lr_rows(rows):
	# Merge L/R rows by (link_start, chainage_start, link_end, chainage_end)
	merged = {}
	for row in rows:
		key = (row["link_start"], row["chainage_start"], row["link_end"], row["chainage_end"])
		if key not in merged:
			merged[key] = row.copy()
		else:
			for col in TRAFFIC_COLUMNS:
				merged[key][col] += row[col]
	# Remove helper columns
	for v in merged.values():
		v.pop("_link_id", None)
		v.pop("_desc", None)
	return list(merged.values())


def main():
	base = os.path.dirname(__file__)
	rmms_dir = os.path.join(base, "RMMS")
	roads_csv = os.path.join(base, "..", "roads.csv")
	# Read unique road names from roads.csv (first column, skip header)
	road_names = set()
	with open(roads_csv, encoding="utf-8") as f:
		next(f)  # skip header
		for line in f:
			if line.strip():
				road = line.split(",", 1)[0].strip()
				if road:
					road_names.add(road)
	all_rows = []
	for road in sorted(road_names):
		traffic_file = os.path.join(rmms_dir, f"{road}.traffic.htm")
		if os.path.exists(traffic_file):
			try:
				rows = parse_traffic_html(traffic_file)
				merged = merge_lr_rows(rows)
				for row in merged:
					row['road'] = road
				all_rows.extend(merged)
			except Exception as e:
				print(f"Error processing {traffic_file}: {e}")
		else:
			print(f"Warning: {traffic_file} not found.")
	if not all_rows:
		print("No traffic data extracted.")
		return
	df = pd.DataFrame(all_rows)
	# Reorder columns
	cols = ["road", "link_start", "link_end", "chainage_start", "chainage_end"] + TRAFFIC_COLUMNS
	df = df[[c for c in cols if c in df.columns]]
	
	df.to_csv(os.path.join(base, "traffic_data_all_roads.csv"), index=False)

if __name__ == "__main__":
	main()
