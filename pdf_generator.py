from fpdf import FPDF
import datetime

def generate_report(email_text, prediction, confidence, risk_level, probabilities, word_weights, color_map, output_path=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font('Arial', 'B', 22)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 12, 'PHISHDETECT AI', 0, 1, 'C')
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, f'Report: {datetime.datetime.now().strftime("%B %d, %Y at %H:%M:%S")}', 0, 1, 'C')
    pdf.ln(8)

    hex_c = color_map[prediction]['hex'].lstrip('#')
    r, g, b = int(hex_c[0:2],16), int(hex_c[2:4],16), int(hex_c[4:6],16)
    if prediction == 'legitimate':
        bg = (230, 255, 230)
    elif prediction == 'traditional_phishing':
        bg = (230, 240, 255)
    else:
        bg = (255, 230, 230)
    pdf.set_fill_color(*bg)
    pdf.set_draw_color(r, g, b)
    pdf.rect(10, pdf.get_y(), 190, 30, 'DF')
    pdf.set_xy(15, pdf.get_y()+5)
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 8, f'Prediction: {prediction.upper().replace("_", " ")}', 0, 1)
    pdf.set_x(15)
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 7, f'Confidence: {confidence:.2f}%  |  Risk: {risk_level}', 0, 1)
    pdf.set_y(pdf.get_y()+10)
    pdf.ln(3)

    pdf.set_font('Arial', 'B', 12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(0, 8, 'LIME Explanation:', 0, 1)
    for word, weight in word_weights[:10]:
        icon = '+' if weight > 0 else '-'
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(0 if weight>0 else 200, 100 if weight>0 else 0, 0)
        pdf.cell(0, 6, f'  {icon} "{word}": {weight:.4f}', 0, 1)

    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, 'Automated analysis by ML model.', 0, 1, 'C')
    if output_path is None:
        output_path = 'phishing_report.pdf'
    pdf.output(output_path)
    return output_path
